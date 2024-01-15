#!/bin/bash
#SBATCH --job-name=event-prediction                            # Specify a name for your job
#SBATCH --output=slurm-logs/out-event-prediction-%j.log        # Specify the output log file
#SBATCH --error=slurm-logs/err-event-prediction-%j.log         # Specify the error log file
#SBATCH --nodes=1                                             # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1                                     # Number of CPU cores per task
#SBATCH --time=24:00:00                                       # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default                                         # Specify the partition (queue) you want to use
#SBATCH --gres=gpu:rtxa5000:1                                 # Number of GPUs per node
#SBATCH --mem=32G                                             # Memory per node


# Project setup
proj_root="/vulcanscratch/mhoover4/code/event-prediction"

# Load any required modules or activate your base environment here if necessary
source $proj_root/.venv/bin/activate

python $proj_root/process_data.py data=ibm_fraud_transaction_med
python $proj_root/pretrain.py     data=ibm_fraud_transaction_med

# python $proj_root/process_data.py data=ibm_fraud_transaction
# python $proj_root/pretrain.py     data=ibm_fraud_transaction