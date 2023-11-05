import torch
import pandas as pd
from transformers import T5Tokenizer
from datasets import Dataset


def tokenize(dataset, tokenizer, max_len=256):
    tokenized_input = tokenizer(
        dataset["reference"], 
        max_length=max_len, 
        truncation=True,
        return_attention_mask = True,
        )

    tokenized_labels = tokenizer(
        text_target=dataset["translation"],
        max_length=max_len,
        truncation=True,
        return_attention_mask = True,
        )

    tokenized_input["labels"] = tokenized_labels["input_ids"]
    return tokenized_input


def make_dataset(data_path, tokenizer=None, max_len=256, return_num=-1):
    data = pd.read_csv(data_path, index_col=0)

    if return_num != -1:
        data = data.head(return_num)
        
    dataset = Dataset.from_pandas(data)
    if tokenizer is not None:
        dataset = dataset.map(lambda x: tokenize(x, tokenizer, max_len=max_len))

    
    return dataset

