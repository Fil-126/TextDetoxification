from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, RobertaForSequenceClassification, RobertaTokenizer
from src.data.make_dataset import make_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm


def predict():
    model = AutoModelForSeq2SeqLM.from_pretrained("models/t5_detox/checkpoint-1000")
    tokenizer = AutoTokenizer.from_pretrained("models/t5_detox/checkpoint-1000")

    dataset = make_dataset("data/interim/test.csv", return_num=3000)


    # Using pretrained toxicity classification model to choose less toxic result from detoxification model outputs
    toxicity_classifier = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
    toxicity_tokenizer =  RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')

    results = []

    for item in tqdm(dataset):
        raw_outputs = model.generate(
            tokenizer.encode(item["reference"], return_tensors="pt"),
            num_beams=10, num_return_sequences=10, max_length=64
        )

        outputs = [tokenizer.decode(out, skip_special_tokens=True) for out in raw_outputs]

        toxicity = [toxicity_classifier(toxicity_tokenizer.encode(out, return_tensors="pt"))[0][0][1] for out in outputs]
        toxicity = [x.detach().numpy() for x in toxicity]

        results.append(outputs[np.argmin(toxicity)])

    series = pd.Series(results)
    series.to_csv("data/interim/predictions.csv")

    