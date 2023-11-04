import pandas as pd
import os
import argparse


def swap_toxic(reference, translation, ref_tox, trn_tox):
    if trn_tox > ref_tox:
        reference, translation = translation, reference

    return pd.Series([reference, translation], index=["reference", "translation"])


def preprocess(data_path, sep="\t", save_path="data/interim/"):
    data = pd.read_csv(data_path, sep=sep)

    preprocessed = data[["reference", "translation", "ref_tox", "trn_tox"]].apply(
        lambda row: swap_toxic(*row), axis=1
        )

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    preprocessed.to_csv(save_path + data_path.split("/")[-1], sep=sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from url")
    parser.add_argument("--path",
                        default= "data/raw/filtered.tsv",
                        dest= "data_path",
                        help= "path to raw data")
    parser.add_argument("--dst",
                        dest= "save_path",
                        default= "data/raw/",
                        help= "save path (default: data/interim/)")
    parser.add_argument("--sep",
                        dest= "sep",
                        default= "\t",
                        help= "csv separator (default: tab)")
    
    args = parser.parse_args()
    preprocess(args.data_path, args.sep, args.save_path)

