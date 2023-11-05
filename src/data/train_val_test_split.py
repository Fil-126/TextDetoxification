import pandas as pd
import numpy as np
import random
import argparse


def split(data_path="data/interim/filtered.tsv", sep="\t", ratio=[0.7, 0.1, 0.2]):
    random.seed(420)
    np.random.seed(420)
    data = pd.read_csv(data_path, sep=sep)[["reference", "translation"]]
    n = len(data)

    indexes = np.random.permutation(n)

    train_val_border = int(ratio[0] * n)
    val_test_border = int((ratio[0] + ratio[1]) * n)
    
    train_indexes = indexes[:train_val_border]
    val_indexes = indexes[train_val_border : val_test_border]
    test_indexes = indexes[val_test_border:]

    train = data.loc[train_indexes]
    val = data.loc[val_indexes]
    test = data.loc[test_indexes]

    train.to_csv("data/interim/train.csv")
    val.to_csv("data/interim/val.csv")
    test.to_csv("data/interim/test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from url")
    parser.add_argument("--path",
                        default= "data/interim/filtered.tsv",
                        dest= "data_path",
                        help= "path to data")
    parser.add_argument("--ratio",
                        dest= "ratio",
                        default= "[0.7,0.1,0.2]",
                        help= "train/val/test split ratio (default: [0.7,0.1,0.2]")
    parser.add_argument("--sep",
                        dest= "sep",
                        default= "\t",
                        help= "csv separator (default: tab)")
    
    args = parser.parse_args()
    split(data_path=args.data_path, sep=args.sep, ratio=eval(args.ratio))
