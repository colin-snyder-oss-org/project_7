# src/training/train.py
import torch
from torch.optim import Adam
from src.models.gnn_model import ComplexGNNModel
from src.data.data_loader import get_data_loader
from src.utils.metrics import compute_accuracy
from contextlib import contextmanager
import traceback

class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = ComplexGNNModel(config).to(config.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.data_loader = get_data_loader(config)
        self.epoch = 0

    @contextmanager
    def exception_handler(self):
        try:
            yield
        except Exception as e:
            print(f"An exception occurred during training: {e}")
            traceback.print_exc()

    def train(self):
        with self.exception_handler():
            for epoch in range(self.config.num_epochs):
                self.model.train()
                total_loss = 0
                for data in self.data_loader:
                    data = data.to(self.config.device)
                    self.optimizer.zero_grad()
                    out = self.model(data.x, data.edge_index)
                    loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch+1}, Loss: {total_loss/len(self.data_loader):.4f}")
                self.epoch += 1
