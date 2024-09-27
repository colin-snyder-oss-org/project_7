# src/main.py
import sys
import importlib
from config import Config

def main():
    config = Config()
    module_name = 'training.train' if sys.argv[1] == 'train' else 'training.evaluate'
    module = importlib.import_module(module_name)
    class_name = 'Trainer' if sys.argv[1] == 'train' else 'Evaluator'
    ClassRef = getattr(module, class_name)
    instance = ClassRef(config)
    getattr(instance, sys.argv[1])()

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ['train', 'evaluate']:
        print("Usage: python main.py [train|evaluate]")
        sys.exit(1)
    main()
