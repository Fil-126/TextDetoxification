from transformers import Seq2SeqTrainer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer
from src.data.make_dataset import make_dataset
import torch
import numpy as np
import random


def train():
    random.seed(420)
    np.random.seed(420)
    torch.manual_seed(420)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(420)

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    training_args = Seq2SeqTrainingArguments(
        output_dir="models/t5_detox",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        max_steps=1000,
        save_steps=100,
        eval_steps=100,
        save_total_limit=3,
        fp16=True,
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=make_dataset("data/interim/train.csv", tokenizer=tokenizer),
        eval_dataset=make_dataset("data/interim/test.csv", tokenizer=tokenizer),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    