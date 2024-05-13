# movies
#python download_and_save_data.py data=amazon_movies_5core
#python train_tokenizer.py data=amazon_movies_5core
#python tokenize_dataset.py data=amazon_movies_5core tokenizer_name=amazon_movies_5core_simple
#python pretrain_HF.py seed=42 name=amazon_movies_reverse wandb=default wandb.tags=['amzn','movies','pretrain'] experiment=amazon_movies_default model.batch_size=64 model.epochs=10 model.train_test_split=.02 model.seq_length=10 model.lr=.00001 impl.print_loss_every_nth_step=1000 model.randomize_order=True impl.save_intermediate_checkpoints=True
#python pretrain_HF.py seed=42 name=amazon_movies wandb=default wandb.tags=['amzn','movies','pretrain'] experiment=amazon_movies_default model.batch_size=64 model.epochs=10 model.train_test_split=.02 model.seq_length=10 model.lr=.00001 impl.print_loss_every_nth_step=1000 model.randomize_order=False impl.save_intermediate_checkpoints=True
python eval.py seed=42 name=amazon_movies experiment=amazon_movies_default model.batch_size=64 model.seq_length=10 model_save_name="amazon_movies_2024-05-10-13-51" model.randomize_order=True
#python eval.py seed=42 name=amazon_movies_reverse experiment=amazon_movies_default model.batch_size=64 model.seq_length=10 model_save_name="amazon_movies_reverse_2024-05-10-14-10" model.randomize_order=True

#python eval.py seed=42 name=amazon_movies_consolidate consolidate=True experiment=amazon_movies_default model.batch_size=64 model.seq_length=10 model_save_name="amazon_movies_2024-05-10-13-51" model.randomize_order=True
#python eval.py seed=42 name=amazon_movies_reverse_consolidate consolidate=True experiment=amazon_movies_default model.batch_size=64 model.seq_length=10 model_save_name="amazon_movies_reverse_2024-05-10-14-10" model.randomize_order=True

python eval.py seed=42 name=amazon_movies experiment=amazon_movies_default model.batch_size=64 model.seq_length=10 model_save_name="amazon_movies_2024-05-10-13-51" model.randomize_order=False
#python eval.py seed=42 name=amazon_movies_reverse experiment=amazon_movies_default model.batch_size=64 model.seq_length=10 model_save_name="amazon_movies_reverse_2024-05-10-14-10" model.randomize_order=False

#python eval.py seed=42 name=amazon_movies_consolidate consolidate=True experiment=amazon_movies_default model.batch_size=64 model.seq_length=10 model_save_name="amazon_movies_2024-05-10-13-51" model.randomize_order=False
#python eval.py seed=42 name=amazon_movies_reverse_consolidate consolidate=True experiment=amazon_movies_default model.batch_size=64 model.seq_length=10 model_save_name="amazon_movies_reverse_2024-05-10-14-10" model.randomize_order=False

#small test
#python train_tokenizer.py data=amazon_movies_5core data.size="_small"
#python tokenize_dataset.py data=amazon_movies_5core tokenizer_name=amazon_movies_5core_small_simple data.size="_small"
#python pretrain_HF.py wandb=default wandb.tags=['amzn','movies','pretrain','small'] experiment=amazon_movies_small model.batch_size=32 model.epochs=100 model.train_test_split=.1 model.seq_length=20 model.lr=.00001 impl.print_loss_every_nth_step=1
#python eval.py data=amazon_movies_5core tokenizer_name=amazon_movies_5core_simple tokenized_data_name=amazon_movies_5core model.state_dict_path="models/Big_MODEL.pth"


# electronics
#python download_and_save_data.py data=amazon_electronics_5core
#python train_tokenizer.py data=amazon_electronics_5core
#python tokenize_dataset.py data=amazon_electronics_5core tokenizer_name=amazon_electronics_5core_simple
#python pretrain_HF.py wandb=default wandb.tags=['amzn','electronics','pretrain'] experiment=amazon_electronics_default model.batch_size=64 model.epochs=10 model.train_test_split=.02 model.seq_length=10 model.lr=.00001 impl.print_loss_every_nth_step=1000


#small test
#python download_and_save_data.py data=amazon_electronics_5core
#python train_tokenizer.py data=amazon_electronics_5core data.size="_small"
#python tokenize_dataset.py data=amazon_electronics_5core tokenizer_name=amazon_electronics_5core_small_simple data.size="_small"
#python pretrain_HF.py experiment=amazon_electronics_small model.batch_size=32 model.epochs=100 model.train_test_split=.1 model.seq_length=20 model.lr=.00001 impl.print_loss_every_nth_step=1
#python eval.py data=amazon_electronics_5core tokenizer_name=amazon_electronics_5core_simple tokenized_data_name=amazon_electronics_5core model.state_dict_path="models/Big_MODEL.pth"
