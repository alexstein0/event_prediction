# Event Prediction

## Download and tokenize data
```
pip install -r requirements.txt
python process_data.py
```

Running the line above loads a small subset of the dataset for testing. To load the full dataset run:
```
python process_data.py data=ibm_fraud_transaction
```

By default we download the dataset as stream in memory to save time reading and writing to disk. If you want to save the tar.gz file of the dataset to disk, run:
```
python process_data.py save_tar=True
```

If you want to save a csv version of the dataset to disk, run:
```
python process_data.py save_csv=True
```

To get a quick look at the dataset, open `dataset_visualization.ipynb` 

## Conduct Pretraining

```
python pretrain.py
```