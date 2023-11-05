from src.models.train import train
from src.models.predict import predict
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training or prediction")
    parser.add_argument("--predict",
                        dest=predict,
                        action="store_true",
                        help= "use for prediction, omit for training")

    args = parser.parse_args()

    if args.predict:
        predict()
    else:
        train()
    