## Text De-toxification

Author: Grigorii Fil (BS21-DS02)
g.fil@innopolis.university

### How to run
To download data, use the following command:
    
    python src/data/download.py

To preprocess raw data, use:

    python src/data/preprocess_raw.py

To make train/val/test split, use:

    python src/data/train_val_test.py


Above commands will generate data in /data folder. After that, you can start training by:

    python main.py

or you can make predictions (will be saved to data/interim/predictions.csv):

    python main.py --predict
