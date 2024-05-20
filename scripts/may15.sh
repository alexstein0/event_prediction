#python download_and_save_data.py data=amazon_movies_5core
#python train_tokenizer.py data=amazon_movies_5core
#python tokenize_dataset.py data=amazon_movies_5core tokenizer_name=amazon_movies_5core_simple
#python pretrain_HF.py name=amazon_movies wandb=default wandb.tags=['amzn','movies','pretrain'] experiment=amazon_movies_default model.batch_size=64 model.epochs=10 model.train_test_split=.02 model.seq_length=10 model.lr=.00001 impl.print_loss_every_nth_step=1000 impl.save_intermediate_checkpoints=True
#python pretrain_HF.py name=amazon_movies_exclude_label wandb=default wandb.tags=['amzn','movies','pretrain'] experiment=amazon_movies_default model.batch_size=64 model.epochs=10 model.train_test_split=.02 model.seq_length=10 model.lr=.00001 impl.print_loss_every_nth_step=1000 impl.save_intermediate_checkpoints=True model.include_labels_in_input=False
#python pretrain_HF.py name=amazon_movies_auc_last wandb=default wandb.tags=['amzn','movies','pretrain'] experiment=amazon_movies_default model.batch_size=64 model.epochs=10 model.train_test_split=.02 model.seq_length=10 model.lr=.00001 impl.print_loss_every_nth_step=1000 impl.save_intermediate_checkpoints=True model.metric_calc_mode=last
#python pretrain_HF.py name=amazon_movies_exclude_label_auc_last wandb=default wandb.tags=['amzn','movies','pretrain'] experiment=amazon_movies_default model.batch_size=64 model.epochs=10 model.train_test_split=.02 model.seq_length=10 model.lr=.00001 impl.print_loss_every_nth_step=1000 impl.save_intermediate_checkpoints=True model.include_labels_in_input=False model.metric_calc_mode=last
#python eval.py seed=42 name=amazon_movies experiment=amazon_movies_default model.batch_size=64 model.seq_length=10 model_save_name="amazon_movies_2024-05-10-13-51" model.randomize_order=True

# small
#python train_tokenizer.py data=amazon_movies_5core data.size="_small"
#python tokenize_dataset.py data=amazon_movies_5core tokenizer_name=amazon_movies_5core_small_simple data.size="_small"
#python pretrain_HF.py name=amazon_movies_small wandb=default wandb.tags=['amzn','movies','pretrain','small'] experiment=amazon_movies_small model.batch_size=64 model.epochs=20 model.train_test_split=.1 model.seq_length=10 model.lr=.00001 impl.print_loss_every_nth_step=100 model.randomize_order=False
#python eval.py seed=42 name=amazon_movies experiment=amazon_movies_small data.size="_small" tokenizer_name=amazon_movies_5core_small_simple model.batch_size=64 model.seq_length=10 model_save_name="amazon_movies_small_2024-05-13-16-33" model.randomize_order=True


# electronics
#python download_and_save_data.py data=amazon_electronics_5core
#python train_tokenizer.py data=amazon_electronics_5core
#python tokenize_dataset.py data=amazon_electronics_5core tokenizer_name=amazon_electronics_5core_simple
#python pretrain_HF.py name=amazon_electronics wandb=default wandb.tags=['amzn','electronics','pretrain'] experiment=amazon_electronics_default model.batch_size=64 model.epochs=10 model.train_test_split=.02 model.seq_length=10 model.lr=.00001 impl.print_loss_every_nth_step=1000 impl.save_intermediate_checkpoints=True
#python pretrain_HF.py name=amazon_electronics_exclude_label wandb=default wandb.tags=['amzn','electronics','pretrain'] experiment=amazon_electronics_default model.batch_size=64 model.epochs=10 model.train_test_split=.02 model.seq_length=10 model.lr=.00001 impl.print_loss_every_nth_step=1000 impl.save_intermediate_checkpoints=True model.include_labels_in_input=False
#python pretrain_HF.py name=amazon_electronics_auc_last wandb=default wandb.tags=['amzn','electronics','pretrain'] experiment=amazon_electronics_default model.batch_size=64 model.epochs=10 model.train_test_split=.02 model.seq_length=10 model.lr=.00001 impl.print_loss_every_nth_step=1000 impl.save_intermediate_checkpoints=True model.metric_calc_mode=last
#python pretrain_HF.py name=amazon_electronics_exclude_label_auc_last wandb=default wandb.tags=['amzn','electronics','pretrain'] experiment=amazon_electronics_default model.batch_size=64 model.epochs=10 model.train_test_split=.02 model.seq_length=10 model.lr=.00001 impl.print_loss_every_nth_step=1000 impl.save_intermediate_checkpoints=True model.include_labels_in_input=False model.metric_calc_mode=last
