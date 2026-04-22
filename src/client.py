"""
Federated Learning Client using Flower Framework
Implements FedAvg for baseline and supports gradient extraction for attacks
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict
from flwr.client import Client, ClientApp
from flwr.common import NDArrays, Scalar
import flwr.common as fc
import numpy as np

class FlowerClient(Client):
    """
    Federated Learning Client for Category 1: Privacy Attacks
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 0.01,
        local_epochs: int = 5
    ):
        """
        Initialize FL Client
        
        Args:
            client_id: Client identifier
            model: Neural network model
            train_loader: Training data loader
            test_loader: Test data loader
            device: Compute device (cuda/cpu)
            learning_rate: Local SGD learning rate
            local_epochs: Number of local epochs per round
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=0.0
        )
        
        # For gradient extraction (attack simulation)
        self.last_gradients: Dict = {}
        self.last_batch_size = 0
        
    def get_parameters(self, config: Dict) -> NDArrays:
        """Get model parameters"""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from server"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
        """
        Train the model locally for one round
        
        Returns:
            updated_parameters: New model parameters
            num_examples: Number of training examples
            metrics: Local training metrics
        """
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Local training
        train_loss = 0.0
        train_acc = 0.0
        num_examples = 0
        
        self.model.train()
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                # Store batch info for potential gradient extraction
                self.last_batch_size = len(x)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                # Backward pass
                loss.backward()
                
                # Store gradients (for attack simulation)
                self._store_gradients()
                
                # Update parameters
                self.optimizer.step()
                
                # Metrics
                epoch_loss += loss.item() * len(x)
                _, predicted = outputs.max(1)
                correct += predicted.eq(y).sum().item()
                total += len(y)
                num_examples += len(x)
            
            train_loss = epoch_loss / total
            train_acc = correct / total
            
            print(f"Client {self.client_id} | Epoch {epoch+1}/{self.local_epochs} | "
                  f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        
        return (
            self.get_parameters(config),
            num_examples,
            {
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "num_examples": num_examples
            }
        )
    
    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate model on test set
        """
        self.set_parameters(parameters)
        
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss += self.criterion(outputs, y).item() * len(x)
                _, predicted = outputs.max(1)
                correct += predicted.eq(y).sum().item()
                total += len(y)
        
        loss /= total
        accuracy = correct / total
        
        return loss, total, {"accuracy": accuracy}
    
    def _store_gradients(self) -> None:
        """Store gradients for attack simulation"""
        self.last_gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.last_gradients[name] = param.grad.clone().detach()
    
    def get_gradients(self) -> Dict:
        """Return stored gradients (for attack analysis)"""
        return self.last_gradients


def create_client_fn(
    model_fn,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cuda",
    learning_rate: float = 0.01,
    local_epochs: int = 5
):
    """Factory function to create FL clients"""
    def client_fn(cid: str) -> FlowerClient:
        model = model_fn()
        return FlowerClient(
            client_id=int(cid),
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            learning_rate=learning_rate,
            local_epochs=local_epochs
        )
    return client_fn