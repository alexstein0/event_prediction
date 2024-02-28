# Event Prediction

## Setup
1. Clone and install dependencies
   ```
   pip install -r requirements.txt
   ```
   If you are running on Windows or have trouble installing pip requirements, run `pip install -r requirements_flexible.txt`.

## Run example
1. Test that it is working by running a training script on 1000 examples. The tokenizer is already created and in the repo.
   ```
   python pretrain_HF.py data=ibm_fraud_transaction data.size=_small tokenizer=simple 
   ```

## Full run
1. Create tokenizer:
    ```
    python process_data_HF.py data=ibm_fraud_transaction tokenizer=simple 
    ```

2. Run training:
    ```
    python pretrain_HF.py data=ibm_fraud_transaction tokenizer=simple 
    ```


By default we download the dataset as stream in memory to save time reading and writing to disk. If you want to save the tar.gz file of the dataset to disk, run:
```
python process_data_HF.py save_tar=True
```

If you want to save a csv version of the dataset to disk, run:
```
python process_data_HF.py save_csv=True
```

To get a quick look at the dataset, open `dataset_visualization.ipynb`
