#python pretrain_HF.py name=lr_experiment_0001 data=ibm_fraud_transaction tokenizer=simple tokenizer_name="ibm_fraud_transaction_simple" tokenized_data_name=ibm_fraud_transaction model.batch_size=32 model.epochs=1 impl.run_eval_every_nth_step=10000 impl.print_loss_every_nth_step=1000 model.lr=.0001
#[2024-05-02 10:19:52,270] Training for epoch: 0     | avg/step: 0:00:00.46 loss:  0.9667 acc: 99.91%  auc:  0.9361 Mem (VRAM/RAM): 12.4150GB/47.2875GB
#[2024-05-02 10:19:52,272] Eval for epoch: 0         | Total dur: 9:48:25.74 Epoch dur: 9:48:25.74 loss:  1.6888 acc: 99.96%  auc:  0.9761 Mem (VRAM/RAM): 12.4150GB/47.2878GB

#python pretrain_HF.py name=lr_experiment_00001 data=ibm_fraud_transaction tokenizer=simple tokenizer_name="ibm_fraud_transaction_simple" tokenized_data_name=ibm_fraud_transaction model.batch_size=32 model.epochs=1 impl.run_eval_every_nth_step=10000 impl.print_loss_every_nth_step=1000 model.lr=.00001
#[2024-04-27 00:35:34,311] Running Eval After Epoch: 0    | batches: 1582   loss:  1.9342   accuracy:  0.9993 auc:  0.9748    Memory (VRAM/RAM): 14.5809/50.1982

#python pretrain_HF.py name=lr_experiment_00005 data=ibm_fraud_transaction tokenizer=simple tokenizer_name="ibm_fraud_transaction_simple" tokenized_data_name=ibm_fraud_transaction model.batch_size=32 model.epochs=1 impl.run_eval_every_nth_step=10000 impl.print_loss_every_nth_step=1000 model.lr=.00005
#[2024-04-28 06:12:58,429] Training for epoch: 0          | avg/step: 0:00:00.249244 loss:  1.0090 acc: 99.91%  auc:  0.9304 Mem (VRAM/RAM): 14.5811GB/45.7839GB
#[2024-04-28 06:12:58,431] Eval for epoch: 0              | Total dur: 5:12:35.305302 Epoch dur: 5:12:35.305301 loss:  1.7020 acc: 99.94%  auc:  0.9685 Mem (VRAM/RAM): 14.5811GB/45.7863GB

#python pretrain_HF.py name=lr_experiment_00008 data=ibm_fraud_transaction tokenizer=simple tokenizer_name="ibm_fraud_transaction_simple" tokenized_data_name=ibm_fraud_transaction model.batch_size=32 model.epochs=1 impl.run_eval_every_nth_step=10000 impl.print_loss_every_nth_step=1000 model.lr=.00008
#[2024-04-28 06:17:01,234] Training for epoch: 0          | avg/step: 0:00:00.252628 loss:  0.9741 acc: 99.91%  auc:  0.9341 Mem (VRAM/RAM): 14.5804GB/45.7742GB
#[2024-04-28 06:17:01,236] Eval for epoch: 0              | Total dur: 5:16:39.662995 Epoch dur: 5:16:39.662992 loss:  1.7644 acc: 99.95%  auc:  0.9616 Mem (VRAM/RAM): 14.5804GB/45.7763GB

#python pretrain_HF.py name=lr_experiment_00002 data=ibm_fraud_transaction tokenizer=simple tokenizer_name="ibm_fraud_transaction_simple" tokenized_data_name=ibm_fraud_transaction model.batch_size=32 model.epochs=1 impl.run_eval_every_nth_step=10000 impl.print_loss_every_nth_step=1000 model.lr=.00002
#[2024-04-28 06:17:31,248] Training for epoch: 0          | avg/step: 0:00:00.250018 loss:  1.1546 acc: 99.89%  auc:  0.8644 Mem (VRAM/RAM): 14.5830GB/45.9204GB
#[2024-04-28 06:17:31,250] Eval for epoch: 0              | Total dur: 5:14:12.800455 Epoch dur: 5:14:12.800454 loss:  1.8559 acc: 99.91%  auc:  0.9607 Mem (VRAM/RAM): 14.5830GB/45.9220GB

#python pretrain_HF.py name=seq_length20 data=ibm_fraud_transaction tokenizer=simple tokenizer_name="ibm_fraud_transaction_simple" tokenized_data_name=ibm_fraud_transaction model.batch_size=16 model.epochs=1 impl.run_eval_every_nth_step=10000 impl.print_loss_every_nth_step=1000 model.lr=.0001 model.seq_length=20
#[2024-04-28 17:10:51,913] Training for epoch: 0 | avg/step: 0:00:00.27 loss:  0.9323 acc: 99.89%  auc:  0.9223 Mem (VRAM/RAM): 15.4661GB/50.3210GB
#[2024-04-28 17:10:51,914] Eval for epoch: 0    | Total dur: 5:44:14.09 Epoch dur: 5:44:14.09 loss:  1.7034 acc: 99.86%  auc:  0.9789 Mem (VRAM/RAM): 15.4661GB/50.3212GB

#python pretrain_HF.py name=seq_length5 data=ibm_fraud_transaction tokenizer=simple tokenizer_name="ibm_fraud_transaction_simple" tokenized_data_name=ibm_fraud_transaction model.batch_size=32 model.epochs=1 impl.run_eval_every_nth_step=10000 impl.print_loss_every_nth_step=1000 model.lr=.0001 model.seq_length=5
#[2024-04-28 10:04:56,883] Training for epoch: 0          | avg/step: 0:00:00.211754 loss:  0.8994 acc: 99.93%  auc:  0.9602 Mem (VRAM/RAM): 8.1764GB/80.4114GB
#[2024-04-28 10:04:56,887] Eval for epoch: 0              | Total dur: 8:50:32.522708 Epoch dur: 8:50:32.522707 loss:  1.8374 acc: 99.96%  auc:  0.9834 Mem (VRAM/RAM): 8.1764GB/77.9260GB

python pretrain_HF.py name=seq_length1 data=ibm_fraud_transaction tokenizer=simple tokenizer_name="ibm_fraud_transaction_simple" tokenized_data_name=ibm_fraud_transaction model.batch_size=32 model.epochs=1 impl.run_eval_every_nth_step=10000 impl.print_loss_every_nth_step=1000 model.lr=.0001 model.seq_length=1