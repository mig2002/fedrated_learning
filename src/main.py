#!/usr/bin/env python3
"""
Main script to run federated learning experiments with GradInversion attacks
Category 1: Privacy & Inference Attacks
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import yaml

from data import create_federated_datasets
from model import get_model
from attack_gradinversion import GradInversionAttack, extract_gradients_from_batch
from utils import (
    calculate_reconstruction_mse,
    calculate_attack_success_rate,
    plot_attack_success_rate,
    save_results_table
)

def run_gradinversion_experiments(args):
    """Run GradInversion attack experiments"""
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading {args.dataset}...")
    client_loaders, test_loader = create_federated_datasets(
        dataset_name=args.dataset,
        num_clients=args.clients,
        alpha=0.1,
        batch_size=32,
        seed=42
    )
    
    # Create model
    model = get_model("ResNet18", num_classes=10, dataset=args.dataset)
    model.to(device)
    
    # Initialize attack
    attack = GradInversionAttack(model, args.config, device=device)
    
    print(f"\nRunning GradInversion attacks on batch sizes: {attack.config['target_batch_sizes']}")
    
    results = []
    
    for batch_size in attack.config['target_batch_sizes']:
        print(f"\n{'='*60}")
        print(f"Testing Batch Size: {batch_size}")
        print(f"{'='*60}")
        
        success_rates = []
        mses = []
        
        # Sample a batch
        data_iter = iter(client_loaders[0])
        try:
            batch_data, batch_labels = next(data_iter)
        except StopIteration:
            print(f"Insufficient data for batch size {batch_size}")
            continue
        
        if len(batch_data) < batch_size:
            batch_data = batch_data[:batch_size]
            batch_labels = batch_labels[:batch_size]
        else:
            batch_data = batch_data[:batch_size]
            batch_labels = batch_labels[:batch_size]
        
        # Extract gradients
        target_gradients = extract_gradients_from_batch(
            model, batch_data, batch_labels, device
        )
        
        # Execute attack
        reconstructed, metrics = attack.attack(
            target_gradients,
            batch_size=batch_size,
            num_classes=10
        )
        
        # Compute metrics
        batch_data = batch_data.to(device)
        mse = calculate_reconstruction_mse(batch_data, reconstructed)
        success_rate = 100 * calculate_attack_success_rate(batch_data, reconstructed, mse_threshold=0.02)
        
        success_rates.append(success_rate)
        mses.append(mse)
        
        print(f"Attack Success Rate: {success_rate:.2f}%")
        print(f"Reconstruction MSE: {mse:.6f}")
        
        results.append({
            'Batch Size': batch_size,
            'Attack Success Rate (%)': success_rate,
            'Reconstruction MSE': mse,
            'Method': 'GradInversion'
        })
    
    # Save and plot results
    print(f"\n{'='*60}")
    print("Saving results...")
    save_results_table({'results': results})
    plot_attack_success_rate(
        [r['Batch Size'] for r in results],
        [r['Attack Success Rate (%)'] for r in results]
    )
    
    print(f"\n{'='*60}")
    print("Experiment completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GradInversion Privacy Attack Experiments"
    )
    parser.add_argument("--config", type=str, 
                       default="configs/gradinversion_attack.yaml",
                       help="Config file path")
    parser.add_argument("--dataset", type=str, default="CIFAR-10",
                       help="Dataset name")
    parser.add_argument("--clients", type=int, default=10,
                       help="Number of clients")
    
    args = parser.parse_args()
    run_gradinversion_experiments(args)