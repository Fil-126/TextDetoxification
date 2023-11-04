import torch
import pandas as pd
from transformers import T5Tokenizer


class DetoxDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer=None, max_len=256):
        if tokenizer is None:
            tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=max_len, legacy=True)
        self.tokenizer = tokenizer

        self.references = []
        self.translations = []

        for i in range(len(data)):
            reference = data["reference"][i]
            translation = data["translation"][i]

            tokenized_ref = self.tokenizer.encode_plus(
                "paraphrase: " + reference,
                add_special_tokens = True,
                max_length = max_len,
                truncation = True,
                padding = True,
                return_attention_mask = True,
                return_tensors = 'pt',  
            )

            tokenized_trn = self.tokenizer.encode_plus(
                translation,
                add_special_tokens = True,
                max_length = max_len,
                truncation = True,
                padding = True,
                return_attention_mask = True,
                return_tensors = 'pt',  
            )

            self.references.append(tokenized_ref)
            self.translations.append(tokenized_trn)

    
    def __len__(self):
        return len(self.references)
    

    def __getitem__(self, index):
        reference_ids = self.references[index]["input_ids"].squeeze()
        translation_ids = self.translations[index]["input_ids"].squeeze()

        reference_mask = self.references[index]["attention_mask"].squeeze()
        translation_mask = self.translations[index]["attention_mask"].squeeze()

        return {
            "reference_ids": reference_ids,
            "reference_mask": reference_mask, 
            "translation_ids": translation_ids, 
            "translation_mask": translation_mask,
            }


def make_datasets(data_path="data/interim/filtered.tsv", sep="\t", tokenizer=None, max_len=256, train_val_test=[0.7, 0.1, 0.2]):
    global train_dataset, val_dataset, test_dataset
    
    data = pd.read_csv(data_path, sep=sep)

    dataset = DetoxDataset(data, tokenizer=tokenizer, max_len=max_len)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, train_val_test)

    return train_dataset, val_dataset, test_dataset

