from .tool import *

class Teacher(SimCSE):
    """
    A class for distilling ranking knowledge from SimCSE-based models. It is the same as the SimCSE except the features are precomputed and passed to the constructor.
    """

    def __init__(self, model_name_or_path: str = "voidism/diffcse-bert-base-uncased-sts", 
                device: str = None,
                num_cells: int = 100,
                num_cells_in_search: int = 10,
                pooler = "cls"):
        
        super().__init__(model_name_or_path, device, num_cells, num_cells_in_search, pooler)

        self.model = self.model.to(self.device if device is None else device)

    def encode(self, 
                inputs = None,
                device: str = "cuda:0", 
                return_numpy: bool = False,
                normalize_to_unit: bool = False,
                keepdim: bool = False,
                batch_size: int = 128,
                max_length: int = 128) -> Union[ndarray, Tensor]:

        target_device = self.device if device is None else device
        single_sentence = False

        embedding_list = [] 
        with torch.no_grad():
            # total_batch = len(inputs["input_ids"]) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            # for batch_id in tqdm(range(total_batch)):
                # inputs = self.tokenizer(
                #     sentence[batch_id*batch_size:(batch_id+1)*batch_size], 
                #     padding=True, 
                #     truncation=True, 
                #     max_length=max_length, 
                #     return_tensors="pt"
                # )
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            outputs = self.model(**inputs, return_dict=True)
            if self.pooler == "cls":
                embeddings = outputs.pooler_output
            elif self.pooler == "cls_before_pooler":
                embeddings = outputs.last_hidden_state[:, 0]
            else:
                raise NotImplementedError
            # if normalize_to_unit:
            #     embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            embedding_list.append(embeddings)
            embeddings = torch.cat(embedding_list)
        
        # if single_sentence and not keepdim:
        #     embeddings = embeddings[0]
        
        # if return_numpy and not isinstance(embeddings, ndarray):
        #     return embeddings.numpy()

        return embeddings