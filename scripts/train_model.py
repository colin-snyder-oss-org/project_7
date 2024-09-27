# scripts/train_model.py
import sys
import os
sys.path.append('src')

from training.train import Trainer
from config import Config

def main():
    config = Config()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
