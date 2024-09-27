# src/training/evaluate.py
import torch
from src.models.gnn_model import ComplexGNNModel
from src.data.data_loader import get_data_loader
from src.utils.metrics import compute_accuracy
from functools import wraps

def evaluation_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Starting evaluation...")
        result = func(*args, **kwargs)
        print("Evaluation completed.")
        return result
    return wrapper

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.model = ComplexGNNModel(config).to(config.device)
        self.data_loader = get_data_loader(config)

    @evaluation_decorator
    def evaluate(self):
        self.model.eval()
        accuracies = []
        with torch.no_grad():
            for data in self.data_loader:
                data = data.to(self.config.device)
                out = self.model(data.x, data.edge_index)
                acc = compute_accuracy(out[data.test_mask], data.y[data.test_mask])
                accuracies.append(acc)
        avg_acc = sum(accuracies) / len(accuracies)
        print(f"Average Test Accuracy: {avg_acc:.2f}%")
