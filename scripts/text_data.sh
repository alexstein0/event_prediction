#python create_text_dataset.py data=amazon_electronics_5core data.save_name="amazon_electronics_text"
#python create_text_dataset.py data=amazon_movies_5core data.save_name="amazon_movies_text"
#python create_text_dataset.py data=ibm_fraud_transaction data.save_name="ibm_fraud_transaction_text"


python run_llm_over_data.py --dataset_name ibm_fraud_transaction_text --batch_size 16
#python run_llm_over_data.py --dataset_name amazon_movies_text
#python run_llm_over_data.py --dataset_name amazon_electronics_text