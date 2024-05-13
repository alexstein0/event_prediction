#python train_tokenizer.py data=ibm_fraud_transaction
#python tokenize_dataset.py data=ibm_fraud_transaction tokenizer_name="ibm_fraud_transaction_simple"
#python pretrain_HF.py data=ibm_fraud_transaction tokenizer_name="ibm_fraud_transaction_simple" tokenized_data_name=ibm_fraud_transaction model.batch_size=32 model.epochs=1 impl.run_eval_every_nth_step=10000 impl.print_loss_every_nth_step=1000 model.lr=.0001
#python eval.py data=ibm_fraud_transaction tokenizer_name=ibm_fraud_transaction_simple tokenized_data_name=ibm_fraud_transaction model.state_dict_path="models/Big_MODEL.pth"
#python eval.py data=ibm_fraud_transaction tokenizer_name=ibm_fraud_transaction_simple tokenized_data_name=ibm_fraud_transaction


#small test
#python train_tokenizer.py data=ibm_fraud_transaction data.size="_small" data.name=ibm_fraud_transaction_small_fixed
python pretrain_HF.py data=ibm_fraud_transaction data.size="_small" tokenizer_name=ibm_fraud_transaction_small_fixed_simple data.name=ibm_fraud_transaction_small_fixed tokenized_data_name=ibm_fraud_transaction_small_fixed model.batch_size=32 model.epochs=100 model.train_test_split=.1 model.seq_length=20 model.lr=.00001 model.batch_size=2 impl.print_loss_every_nth_step=1
#tokenized_data_name=ibm_fraud_transaction (small)
#python eval.py data=ibm_fraud_transaction tokenizer_name=ibm_fraud_transaction_simple tokenized_data_name=ibm_fraud_transaction model.state_dict_path="models/Big_MODEL.pth"
