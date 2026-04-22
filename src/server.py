"""
Federated Learning Server using Flower Framework
Implements FedAvg aggregation and server-side metrics
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Callable, Optional
from flwr.server import ServerApp, Server
from flwr.server.strategy import FedAvg
from flwr.common import NDArrays, Scalar, Context
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class FedAvgCustom(FedAvg):
    """
    Custom FedAvg with Category 1 (Privacy) tracking
    """
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable[[int, NDArrays, Dict], Optional[Tuple[float, Dict]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[NDArrays] = None,
        fit_metrics_aggregation_fn: Optional[Callable[[List[Tuple[int, Dict]]], Dict]] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable[[List[Tuple[int, Dict]]], Dict]] = None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        
        self.round_history = []
        self.privacy_metrics = {
            "rounds": [],
            "avg_loss": [],
            "avg_accuracy": [],
            "communication_rounds": [],
            "total_communication_mb": 0.0
        }
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[int, NDArrays, Dict]],
        failures: List[Tuple[int, NDArrays, Dict]],
    ) -> Tuple[Optional[NDArrays], Dict]:
        """
        Aggregate fit results and track metrics
        """
        # Call parent aggregation
        aggregated_parameters, metrics_dict = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Track communication cost
        if results:
            num_examples = sum([metrics["num_examples"] for _, _, metrics in results])
            avg_loss = np.mean([metrics["train_loss"] for _, _, metrics in results])
            avg_acc = np.mean([metrics["train_accuracy"] for _, _, metrics in results])
            
            # Compute communication cost (bytes transmitted)
            model_size_mb = sum(p.size for params in aggregated_parameters 
                               for p in params) * 4 / (1024 * 1024)  # 4 bytes per float32
            comm_cost_mb = model_size_mb * len(results) * 2  # ×2 for upload + download
            
            self.privacy_metrics["rounds"].append(server_round)
            self.privacy_metrics["avg_loss"].append(avg_loss)
            self.privacy_metrics["avg_accuracy"].append(avg_acc)
            self.privacy_metrics["communication_rounds"].append(comm_cost_mb)
            self.privacy_metrics["total_communication_mb"] += comm_cost_mb
            
            print(f"\nRound {server_round} Aggregation:")
            print(f"  Avg Train Loss: {avg_loss:.4f}")
            print(f"  Avg Train Acc: {avg_acc:.4f}")
            print(f"  Communication Cost: {comm_cost_mb:.2f} MB")
            print(f"  Total Communication: {self.privacy_metrics['total_communication_mb']:.2f} MB")
        
        return aggregated_parameters, metrics_dict
    
    def save_metrics(self, save_path: str = "results/server_metrics.json"):
        """Save server-side metrics"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        metrics_to_save = {
            "timestamp": datetime.now().isoformat(),
            "privacy_metrics": {
                "rounds": self.privacy_metrics["rounds"],
                "avg_loss": self.privacy_metrics["avg_loss"],
                "avg_accuracy": self.privacy_metrics["avg_accuracy"],
                "communication_rounds": self.privacy_metrics["communication_rounds"],
                "total_communication_mb": self.privacy_metrics["total_communication_mb"]
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        
        print(f"\nServer metrics saved to {save_path}")


class ServerMetricsCollector:
    """Collector for server-side evaluation metrics"""
    
    def __init__(self, test_loader, device: str = "cuda"):
        self.test_loader = test_loader
        self.device = device
        self.metrics_history = []
    
    def evaluate_global_model(
        self,
        server_round: int,
        parameters: NDArrays,
        model: nn.Module,
        config: Dict
    ) -> Tuple[float, Dict]:
        """
        Evaluate global model on test set
        """
        # Load parameters into model
        model.eval()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=self.device) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        # Evaluate
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = model(x)
                loss = criterion(outputs, y)
                total_loss += loss.item() * len(x)
                _, predicted = outputs.max(1)
                correct += predicted.eq(y).sum().item()
                total += len(y)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        metrics = {
            "round": server_round,
            "test_loss": avg_loss,
            "test_accuracy": accuracy,
            "correct": correct,
            "total": total
        }
        
        self.metrics_history.append(metrics)
        
        print(f"\nRound {server_round} Server Evaluation:")
        print(f"  Test Loss: {avg_loss:.4f}")
        print(f"  Test Accuracy: {accuracy:.4f}")
        
        return avg_loss, {"accuracy": accuracy}
    
    def save_metrics(self, save_path: str = "results/evaluation_metrics.json"):
        """Save evaluation history"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
        
        print(f"Evaluation metrics saved to {save_path}")


def create_evaluate_fn(
    model: nn.Module,
    test_loader,
    device: str = "cuda"
):
    """Create evaluation function for server"""
    collector = ServerMetricsCollector(test_loader, device)
    
    def evaluate_fn(server_round: int, parameters: NDArrays, config: Dict):
        return collector.evaluate_global_model(server_round, parameters, model, config)
    
    return evaluate_fn, collector


def get_fit_config(server_round: int):
    """Return fit config for each round"""
    return {
        "local_epochs": 5,
        "learning_rate": 0.01,
        "batch_size": 32,
        "server_round": server_round
    }


def get_evaluate_config(server_round: int):
    """Return evaluate config for each round"""
    return {"server_round": server_round}