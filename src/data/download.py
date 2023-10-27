from urllib.request import urlretrieve
import pathlib
import zipfile
import argparse


def download_data(url, save_path = "data/raw/"):
    path = urlretrieve(url)[0]

    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(save_path)

    pathlib.Path(path).unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from url")
    parser.add_argument("--url",
                        default= "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip",
                        help= "url of the document")
    parser.add_argument("--dst",
                        dest= "save_path",
                        default= "data/raw/",
                        help= "save path (default: data/raw/)")
    
    args = parser.parse_args()
    download_data(args.url, args.save_path)
