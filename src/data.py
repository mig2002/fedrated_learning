"""
Data loading and partitioning utilities for FL experiments
Implements Dirichlet distribution for non-IID partitioning
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from typing import List, Tuple, Optional
import json
from pathlib import Path

class CIFAR10NonIID(Dataset):
    """CIFAR-10 wrapper for non-IID sampling"""
    
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def load_cifar10(data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """Load CIFAR-10 dataset"""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    testset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    return trainset, testset


def load_cifar100(data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """Load CIFAR-100 dataset"""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                           (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                           (0.2675, 0.2565, 0.2761)),
    ])
    
    trainset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    testset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    return trainset, testset


def load_mnist(data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """Load MNIST dataset"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    
    testset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    return trainset, testset


class DirichletPartitioner:
    """Partition dataset using Dirichlet distribution"""
    
    def __init__(
        self,
        dataset: Dataset,
        num_clients: int,
        alpha: float = 0.1,
        num_classes: int = 10,
        seed: int = 42,
        save_partitions: bool = False
    ):
        """
        Args:
            dataset: Full dataset
            num_clients: Number of clients
            alpha: Dirichlet concentration parameter
                  - alpha → 0: more non-IID
                  - alpha = 1.0: IID
            num_classes: Number of classes
            seed: Random seed
            save_partitions: Save partition info to file
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.alpha = alpha
        self.num_classes = num_classes
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Get class indices
        self.class_indices = self._get_class_indices()
        
        # Generate partitions
        self.client_indices = self._generate_dirichlet_partitions()
        
        if save_partitions:
            self._save_partitions()
    
    def _get_class_indices(self) -> List[List[int]]:
        """Get indices for each class"""
        class_indices = [[] for _ in range(self.num_classes)]
        
        if hasattr(self.dataset, 'targets'):
            targets = np.array(self.dataset.targets)
        else:
            targets = np.array([self.dataset[i][1] for i in range(len(self.dataset))])
        
        for idx, label in enumerate(targets):
            class_indices[label].append(idx)
        
        return class_indices
    
    def _generate_dirichlet_partitions(self) -> List[List[int]]:
        """Generate non-IID partitions using Dirichlet distribution"""
        
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet(
            np.repeat(self.alpha, self.num_clients),
            self.num_classes
        )
        
        client_indices = [[] for _ in range(self.num_clients)]
        
        for class_idx in range(self.num_classes):
            class_data = self.class_indices[class_idx]
            np.random.shuffle(class_data)
            
            for client_id in range(self.num_clients):
                num_samples = int(proportions[class_idx, client_id] * len(class_data))
                client_indices[client_id].extend(
                    class_data[:num_samples]
                )
                class_data = class_data[num_samples:]
        
        return client_indices
    
    def _save_partitions(self, save_path: str = "results/partitions.json"):
        """Save partition information"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        partition_info = {
            "num_clients": self.num_clients,
            "alpha": self.alpha,
            "num_classes": self.num_classes,
            "client_sizes": [len(indices) for indices in self.client_indices],
            "total_samples": len(self.dataset)
        }
        
        with open(save_path, 'w') as f:
            json.dump(partition_info, f, indent=4)
    
    def get_client_indices(self, client_id: int) -> List[int]:
        """Get data indices for specific client"""
        return self.client_indices[client_id]
    
    def get_all_indices(self) -> List[List[int]]:
        """Get all client indices"""
        return self.client_indices


def create_federated_datasets(
    dataset_name: str = "CIFAR-10",
    num_clients: int = 10,
    alpha: float = 0.1,
    batch_size: int = 32,
    seed: int = 42,
    data_dir: str = "./data"
) -> Tuple[List[DataLoader], DataLoader, DirichletPartitioner]:
    """
    Create federated datasets with non-IID partitioning
    
    Returns:
        client_loaders: List of DataLoaders for each client
        test_loader: Global test DataLoader
        partitioner: DirichletPartitioner object
    """
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
    if dataset_name == "CIFAR-10":
        trainset, testset = load_cifar10(data_dir)
        num_classes = 10
    elif dataset_name == "CIFAR-100":
        trainset, testset = load_cifar100(data_dir)
        num_classes = 100
    elif dataset_name == "MNIST":
        trainset, testset = load_mnist(data_dir)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create partitioner
    partitioner = DirichletPartitioner(
        trainset,
        num_clients=num_clients,
        alpha=alpha,
        num_classes=num_classes,
        seed=seed,
        save_partitions=True
    )
    
    # Create client loaders
    client_loaders = []
    for client_id in range(num_clients):
        client_indices = partitioner.get_client_indices(client_id)
        client_subset = Subset(trainset, client_indices)
        client_loader = DataLoader(
            client_subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        client_loaders.append(client_loader)
    
    # Create test loader
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    print(f"\nDataset: {dataset_name}")
    print(f"Total clients: {num_clients}")
    print(f"Dirichlet alpha: {alpha}")
    print(f"Client batch sizes: {[len(loader.dataset) for loader in client_loaders]}")
    print(f"Test set size: {len(testset)}")
    
    return client_loaders, test_loader, partitioner