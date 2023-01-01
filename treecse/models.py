import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from geomloss import SamplesLoss

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class JointMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        list_layers = [nn.Linear(2 * config.hidden_size, 2 * config.hidden_size, bias=False),
                       nn.ReLU(inplace=True),
                       nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x, **kwargs):
        return self.net(x)

class ClassifierHead(nn.Module):

    def __init__(self, config, num_classes=17):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, num_classes, bias=False)

    def forward(self, features, **kwargs):
        return self.dense(features)


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Divergence(nn.Module):
    """
    Jensen-Shannon divergence, used to measure ranking consistency between similarity lists obtained from examples with two different dropout masks
    """
    def __init__(self, beta_):
        super(Divergence, self).__init__()
        self.eps = 1e-7
        self.beta_ = beta_

    def forward(self, S, S_prime):
        S_hat = S.softmax(dim=-1)
        S_hat_prime = S_prime.softmax(dim=-1)
        S1 = S_hat * torch.log((2 * S_hat) / (S_hat + S_hat_prime))
        S2 = S_hat_prime * torch.log((2 * S_hat_prime) / (S_hat + S_hat_prime))
        return self.beta_ * (S1 + S2).sum()

class ListNet(nn.Module):
    """
    ListNet objective for ranking distillation; minimizes the cross entropy between permutation [top-1] probability distribution and ground truth obtained from teacher
    """
    def __init__(self, tau, gamma_):
        super(ListNet, self).__init__()
        self.teacher_temp_scaled_sim = Similarity(tau / 2)
        self.student_temp_scaled_sim = Similarity(tau)
        self.gamma_ = gamma_

    def forward(self, student_top1_sim_pred, teacher_top1_sim_pred):
        p = F.log_softmax(student_top1_sim_pred.fill_diagonal_(float('-inf')), dim=-1)
        q = F.softmax(teacher_top1_sim_pred.fill_diagonal_(float('-inf')), dim=-1)
        loss = -(q*p).nansum()  / q.nansum()
        return self.gamma_ * loss 

class ListMLE(nn.Module):
    """
    ListMLE objective for ranking distillation; maximizes the liklihood of the ground truth permutation (sorted indices of the ranking lists obtained from teacher) 
    """
    def __init__(self, tau, gamma_):
        super(ListMLE, self).__init__()
        self.temp_scaled_sim = Similarity(tau)
        self.gamma_ = gamma_ 
        self.eps = 1e-7

    def forward(self, student_top1_sim_pred, teacher_top1_sim_pred):
        # student_top1_sim_pred = self.temp_scaled_sim(z1.unsqueeze(1), z2.unsqueeze(0))

        y_pred = student_top1_sim_pred # .softmax(dim=-1) # .fill_diagonal_(float('-inf')).softmax(dim=-1)
        y_true = teacher_top1_sim_pred # .softmax(dim=-1) # .fill_diagonal_(float('-inf')).softmax(dim=-1)

        # shuffle for randomised tie resolution
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
        mask = y_true_sorted == -1
        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float('-inf')
        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
        observation_loss = torch.log(cumsums + self.eps) - preds_sorted_by_true_minus_max
        observation_loss[mask] = 0.0

        return self.gamma_ * torch.mean(torch.sum(observation_loss, dim=1))

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls" or cls.model_args.pooler_type == "avg":
        cls.mlp = MLPLayer(config)
    if cls.model_args.two_poolers:
        cls.jnt = JointMLP(config)
    if cls.model_args.do_clf:
        cls.clf = ClassifierHead(config, 17)

    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.div = Divergence(beta_=cls.model_args.beta_)
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    zP_zLR_sim_T=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
    xP, xL, xR = torch.split(pooler_output, split_size_or_sections=1, dim=1) # (bs, 1, hidden) x 3

    if not cls.model_args.two_poolers:
        # Tree-based ensembling
        xLR = 0.5 * (xL + xR) # ensemble left and right constituents to create positive example for xi
        pooler_output = torch.cat([xP, xLR], dim=1) # (bs, 2, hidden)

        # If using "cls", we add an extra MLP layer
        # (same as BERT's original implementation) over the representation.
        pooler_output = cls.mlp(pooler_output) # (bs, 2, hidden)

        # Separate representation
        zP, zLR = torch.split(pooler_output, split_size_or_sections=1, dim=1) # (bs, 1, hidden) x 2
        zP, zLR = zP.squeeze(), zLR.squeeze() # (bs, hidden) x 2
    else:
        # Tree-based ensembling
        zP = cls.mlp(xP).squeeze()
        zLR = cls.jnt(torch.cat([xL, xR], dim=2)).squeeze()


    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:

        # Dummy vectors for allgather
        zP_list = [torch.zeros_like(zP) for _ in range(dist.get_world_size())]
        zLR_list = [torch.zeros_like(zLR) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=zP_list, tensor=zP.contiguous())
        dist.all_gather(tensor_list=zLR_list, tensor=zLR.contiguous())
        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        zP_list[dist.get_rank()] = zP
        zLR_list[dist.get_rank()] = zLR
        # Get full batch embeddings: (bs x N, hidden)
        zP = torch.cat(zP_list, 0)
        zLR = torch.cat(zLR_list, 0)

    loss = 0
    zP_zLR_sim_S = cls.sim(zP.unsqueeze(1), zLR.unsqueeze(0))
    zLR_zP_sim_S = cls.sim(zLR.unsqueeze(1), zP.unsqueeze(0))

    # L_infoNCE
    if cls.model_args.do_nce:
        nce_labels = torch.arange(zP_zLR_sim_S.size(0)).long().to(cls.device)
        nce_loss_fct = nn.CrossEntropyLoss()
        loss = loss + nce_loss_fct(zP_zLR_sim_S, nce_labels) 

    # L_clf
    if cls.model_args.do_clf:
        clf_output = cls.clf(zLR)
        class_weights = [0.4696812298220549, 0.35647741811866074, 0.18797756457396197, 1.2128324562464186, 3.4169412949618976, 1.0654509472935543, 0.7755944591390502, 8.480163944971833, 1.3588046813799115, 7.098408449295764, 1.0315564016602632, 0, 15.927678818006076, 41.81936010151364, 70.86361542005837, 51.45168664622247, 2584.8347338935573]
        class_weights = torch.tensor(class_weights, dtype=zP.dtype, device=cls.device)

        scl_loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        labels = labels.squeeze().long().to(cls.device)
        loss = loss + (cls.model_args.delta_ * scl_loss_fct(clf_output, labels))

    # L_distillation
    if cls.model_args.do_kd:
        kd_loss_fct = (ListMLE(cls.model_args.tau2, cls.model_args.gamma_) if cls.model_args.distillation_loss == "listmle" else ListNet(cls.model_args.tau2, cls.model_args.gamma_))
        loss = loss + kd_loss_fct(zP_zLR_sim_S.clone(), zP_zLR_sim_T.clone()) # zP_zLR_sim_T is the similarity matrix computed by the teacher(s)

    # L_consistency
    if cls.model_args.do_sd:
        loss = loss + cls.div(zP_zLR_sim_S.clone().softmax(dim=-1).clamp(min=1e-7), zLR_zP_sim_S.clone().softmax(dim=-1).clamp(min=1e-7)) 

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (zP_zLR_sim_S,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=zP_zLR_sim_S,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )

class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        zP_zLR_sim_T=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                zP_zLR_sim_T=zP_zLR_sim_T,
            )

class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        zP_zLR_sim_T=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                zP_zLR_sim_T=zP_zLR_sim_T,
            )