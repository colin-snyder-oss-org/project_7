# src/utils/metrics.py
import torch
from functools import reduce

def compute_accuracy(pred, target):
    pred_labels = pred.argmax(dim=1)
    correct = (pred_labels == target).sum().item()
    total = target.size(0)
    accuracy = correct / total * 100
    return accuracy

def compute_f1_score(pred, target):
    # Placeholder for complex F1 score calculation
    pass
