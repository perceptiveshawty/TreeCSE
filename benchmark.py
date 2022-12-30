import argparse
import logging
import os
from typing import List, Dict, Tuple, Type, Union

logging.basicConfig(level=logging.INFO)

os.environ["HF_DATASETS_OFFLINE"]="0" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="0" # 1 for offline
os.environ["XDG_CACHE_HOME"]="/scratch/user/chanchanis/.cache"
os.environ["HF_HOME"]="/scratch/user/chanchanis/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"]="/scratch/user/chanchanis/.cache/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"]="/scratch/user/chanchanis/.cache/huggingface/datasets"
os.environ["HF_MODULES_CACHE"]="/scratch/user/chanchanis/.cache/huggingface/modules"
os.environ["HF_METRICS_CACHE"]="/scratch/user/chanchanis/.cache/huggingface/metrics"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from numpy import ndarray
from mteb import MTEB
from transformers import AutoModel, AutoTokenizer
import torch
from torch import Tensor, device
from tqdm import tqdm


TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    "SummEval",
]

# TASK_LIST = TASK_LIST_CLASSIFICATION + TASK_LIST_CLUSTERING + TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_RERANKING + TASK_LIST_RETRIEVAL + TASK_LIST_STS
TASK_LIST = TASK_LIST_STS + TASK_LIST_RETRIEVAL

class StructCSEWrapper:
    """
    A class for embedding sentences, calculating similarities, and retriving sentences by SimCSE.
    """
    def __init__(self, model_name_or_path: str, 
                 device: str = None,
                 num_cells: int = 100,
                 num_cells_in_search: int = 10,
                 pooler = "cls_before_pooler"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.index = None
        self.is_faiss_index = False
        self.num_cells = num_cells
        self.num_cells_in_search = num_cells_in_search

        self.pooler = pooler
    
    def encode(self, sentence: Union[str, List[str]], 
                device: str = None, 
                return_numpy: bool = False,
                normalize_to_unit: bool = False,
                keepdim: bool = False,
                batch_size: int = 128,
                max_length: int = 128,
                show_progress_bar: bool = True,
                **kwargs) -> Union[ndarray, Tensor]:

        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)
        
        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = [] 
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            disable_tqdm = (show_progress_bar == False)
            for batch_id in tqdm(range(total_batch), disable=disable_tqdm):
                inputs = self.tokenizer(
                    sentence[batch_id*batch_size:(batch_id+1)*batch_size], 
                    padding=True, 
                    truncation=True, 
                    max_length=max_length, 
                    return_tensors="pt"
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                outputs = self.model(**inputs, return_dict=True)
                if self.pooler == "cls":
                    embeddings = outputs.pooler_output
                elif self.pooler == "cls_before_pooler":
                    embeddings = outputs.last_hidden_state[:, 0]
                else:
                    raise NotImplementedError
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)
        
        if single_sentence and not keepdim:
            embeddings = embeddings[0]
        
        if return_numpy and not isinstance(embeddings, ndarray):
            return embeddings.numpy()
        return embeddings


class SimCSEWrapper:
    def __init__(self, modelpath="runs/structcse-ensemble-bert-base-uncased"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        self.model = AutoModel.from_pretrained(modelpath).to(self.device)
        self.model.eval()

    def encode(self, sentences, batch_size=256, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            inputs = self.tokenizer(sentences_batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k,v in inputs.items()}
            # Get the embeddings
            with torch.no_grad():
                embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            all_embeddings.extend(embeddings.cpu().numpy())
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        return all_embeddings

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--startid", type=int)
    parser.add_argument("--endid", type=int)
    parser.add_argument("--modelpath", type=str, default="runs/structcse-ensemble-bert-base-uncased")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--taskname", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=128)
    args = parser.parse_args()
    return args

def main(args):

    model = StructCSEWrapper(args.modelpath)

    for task in TASK_LIST:
        print("Running task: ", task)
        eval_splits = ["validation"] if task == "MSMARCO" else ["test"]
        # model_name = args.modelpath.split("/")[-1].split("_")[-1]
        model_name = "structcse-bert-base-uncased"
        evaluation = MTEB(tasks=[task], task_langs=[args.lang], task_types=['STS'])
        evaluation.run(model, output_folder=f"results/{model_name}", batch_size=args.batchsize, eval_splits=eval_splits)

if __name__ == "__main__":
    args = parse_args()
    main(args)