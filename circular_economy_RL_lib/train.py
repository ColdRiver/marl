# Overwrite this file as: /kaggle/working/marl/circular_economy_RL_lib/train.py
import os
import sys

# Prevent OpenMP and MKL thread-pool deadlocks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import multiprocessing
from utils import create_logger
from trainer import BilevelTrainer  # Fixed: Imports BilevelTrainer instead of deprecated Trainer

logger_params = {
    'log_file': {
        'desc': 'bilevel_train_marl',
        'filename': 'run_log'
    }
}

def main():
    create_logger(**logger_params)
    print(f"System CPUs available: {multiprocessing.cpu_count()}")
    print("Initializing Gaur et al. (2025) Bilevel Reinforcement Learning inside marl...")
    
    # Instantiate the hierarchical trainer
    trainer = BilevelTrainer()  # Fixed: Instantiates BilevelTrainer
    
    # Launch training
    trainer.learn()

if __name__ == "__main__":
    main()
