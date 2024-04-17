#python train_tokenizer.py data=ibm_fraud_transaction tokenizer=simple
python pretrain_HF.py data=ibm_fraud_transaction tokenizer=simple save_csv=True tokenizer_name="ibm_fraud_transaction_simple"  preprocess_only=True
#tokenized_data_name=ibm_fraud_transaction (small)
#python eval.py data=ibm_fraud_transaction tokenizer=simple tokenizer_name=ibm_fraud_transaction_simple tokenized_data_name=ibm_fraud_transaction model.state_dict_path="models/Big_MODEL.pth"
#python eval.py data=ibm_fraud_transaction tokenizer=simple tokenizer_name=ibm_fraud_transaction_simple tokenized_data_name=ibm_fraud_transaction


#small test
#python train_tokenizer.py data=ibm_fraud_transaction tokenizer=simple data.size="_small" data.name=ibm_fraud_transaction_small_fixed
#python pretrain_HF.py data=ibm_fraud_transaction tokenizer=simple save_csv=True tokenizer_name="ibm_fraud_transaction_small_fixed_simple" data.size="_small"  data.name=ibm_fraud_transaction_small_fixed #tokenized_data_name="ibm_fraud_transaction"
#tokenized_data_name=ibm_fraud_transaction (small)
#python eval.py data=ibm_fraud_transaction tokenizer=simple tokenizer_name=ibm_fraud_transaction_simple tokenized_data_name=ibm_fraud_transaction model.state_dict_path="models/Big_MODEL.pth"
