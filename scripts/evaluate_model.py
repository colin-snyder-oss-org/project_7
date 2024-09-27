# scripts/evaluate_model.py
import sys
import os
sys.path.append('src')

from training.evaluate import Evaluator
from config import Config

def main():
    config = Config()
    evaluator = Evaluator(config)
    evaluator.evaluate()

if __name__ == "__main__":
    main()
