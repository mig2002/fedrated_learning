"""
Main Federated Learning + GradInversion Experiment
Category 1: Privacy & Inference Attacks
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime

from data import create_federated_datasets
from model import get_model
from client import FlowerClient, create_client_fn
from server import FedAvgCustom, ServerMetricsCollector, create_evaluate_fn, get_fit_config, get_evaluate_config
from attack_gradinversion import GradInversionAttack, extract_gradients_from_batch
from utils import MetricsComputer, ResultsVisualizer, ResultsExporter, create_experiment_summary


class FLExperimentRunner:
    """Main experiment runner for FL + GradInversion"""
    
    def __init__(self, config_path: str):
        """Initialize experiment runner"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_paths()
        
        print(f"[FL Experiment] Device: {self.device}")
        print(f"[FL Experiment] Config loaded from {config_path}")
    
    def setup_paths(self):
        """Create necessary directories"""
        Path("results/plots").mkdir(parents=True, exist_ok=True)
        Path("results/checkpoints").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
    
    def run_baseline_fedavg(self):
        """Run FedAvg baseline"""
        print("\n" + "="*80)
        print("FEDAVG BASELINE EXPERIMENT")
        print("="*80)
        
        config = self.config['fedavg']
        
        # Create datasets
        client_loaders, test_loader, partitioner = create_federated_datasets(
            dataset_name=config['dataset'],
            num_clients=config['num_clients'],
            alpha=config['dirichlet_alpha'],
            batch_size=config['local_batch_size'],
            seed=42
        )
        
        # Create model
        model = get_model("ResNet18", num_classes=10, dataset=config['dataset'])
        
        print(f"\nFedAvg Configuration:")
        print(f"  Dataset: {config['dataset']}")
        print(f"  Clients: {config['num_clients']}")
        print(f"  Rounds: {config['num_rounds']}")
        print(f"  Local Epochs: {config['local_epochs']}")
        print(f"  Batch Size: {config['local_batch_size']}")
        print(f"  Dirichlet α: {config['dirichlet_alpha']}")
        
        # Collect results
        all_accuracies = []
        all_losses = []
        
        for round_idx in range(config['num_rounds']):
            print(f"\n[Round {round_idx + 1}/{config['num_rounds']}]", end=" ")
            
            # Simulate local training (without actual Flower server)
            round_accs = []
            round_losses = []
            
            for client_id in range(min(int(config['num_clients'] * config['client_fraction']), 
                                       config['num_clients'])):
                client = FlowerClient(
                    client_id=client_id,
                    model=model,
                    train_loader=client_loaders[client_id],
                    test_loader=test_loader,
                    device=self.device,
                    learning_rate=config['learning_rate'],
                    local_epochs=config['local_epochs']
                )
                
                params, num_examples, metrics = client.fit(
                    [torch.randn(1)],  # Dummy params
                    {'epoch': round_idx}
                )
                
                round_accs.append(metrics['train_accuracy'])
                round_losses.append(metrics['train_loss'])
            
            avg_acc = np.mean(round_accs)
            avg_loss = np.mean(round_losses)
            
            all_accuracies.append(avg_acc * 100)  # Convert to percentage
            all_losses.append(avg_loss)
            
            print(f"Avg Acc: {avg_acc:.4f}, Avg Loss: {avg_loss:.4f}")
        
        # Save results
        results = {
            'fedavg': {
                'accuracies': all_accuracies,
                'losses': all_losses
            }
        }
        
        with open('results/baseline_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n✓ Baseline results saved!")
        return results
    
    def run_gradinversion_attack(self):
        """Run GradInversion attacks on different batch sizes"""
        print("\n" + "="*80)
        print("GRADINVERSION ATTACK EXPERIMENT")
        print("="*80)
        
        # Create datasets
        client_loaders, test_loader, _ = create_federated_datasets(
            dataset_name="CIFAR-10",
            num_clients=10,
            alpha=0.1,
            batch_size=32,
            seed=42
        )
        
        # Create model
        model = get_model("ResNet18", num_classes=10, dataset="CIFAR-10")
        model.eval()
        
        # Initialize attack
        attack = GradInversionAttack(model, "configs/gradinversion_attack.yaml", self.device)
        
        batch_sizes = [4, 8, 16, 32, 48]
        attack_results = []
        
        print(f"\nAttacking with batch sizes: {batch_sizes}")
        
        for batch_size in batch_sizes:
            print(f"\n{'='*60}")
            print(f"Batch Size: {batch_size}")
            print(f"{'='*60}")
            
            # Get a batch from test set
            data_iter = iter(test_loader)
            try:
                batch_images, batch_labels = next(data_iter)
            except StopIteration:
                print(f"Insufficient test data for batch size {batch_size}")
                continue
            
            # Truncate to batch size
            batch_images = batch_images[:batch_size].to(self.device)
            batch_labels = batch_labels[:batch_size].to(self.device)
            
            # Extract gradients
            target_gradients = extract_gradients_from_batch(
                model, batch_images, batch_labels, self.device
            )
            
            # Run attack
            reconstructed, metrics = attack.attack(
                target_gradients,
                batch_size=batch_size,
                num_classes=10,
                original_images=batch_images
            )
            
            # Compute attack metrics
            mse = MetricsComputer.reconstruction_mse(batch_images, reconstructed)
            psnr = MetricsComputer.reconstruction_psnr(batch_images, reconstructed)
            ssim = MetricsComputer.reconstruction_ssim(batch_images, reconstructed)
            success_rate = MetricsComputer.attack_success_rate(batch_images, reconstructed, mse_threshold=0.02)
            
            attack_results.append({
                'batch_size': batch_size,
                'mse': mse,
                'psnr': psnr,
                'ssim': ssim,
                'attack_success': success_rate,
                'num_seeds': 8
            })
            
            print(f"\nAttack Metrics:")
            print(f"  MSE: {mse:.6f}")
            print(f"  PSNR: {psnr:.4f} dB")
            print(f"  SSIM: {ssim:.4f}")
            print(f"  Success Rate: {success_rate:.0%}")
        
        # Visualize results
        batch_sizes_list = [r['batch_size'] for r in attack_results]
        success_rates = [r['attack_success'] * 100 for r in attack_results]
        
        ResultsVisualizer.plot_attack_success_rate(batch_sizes_list, success_rates)
        
        # Save results
        with open('results/attack_results.json', 'w') as f:
            json.dump(attack_results, f, indent=4)
        
        print(f"\n✓ Attack results saved!")
        return attack_results
    
    def run_full_experiment(self):
        """Run complete FL + Attack experiment"""
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + "  FEDERATED LEARNING + GRADINVERSION PRIVACY ATTACK EXPERIMENT".center(78) + "█")
        print("█" + " "*78 + "█")
        print("█"*80)
        
        experiment_start = datetime.now()
        print(f"\nExperiment started: {experiment_start.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run baseline
        baseline_results = self.run_baseline_fedavg()
        
        # Run attacks
        attack_results = self.run_gradinversion_attack()
        
        experiment_end = datetime.now()
        duration = (experiment_end - experiment_start).total_seconds() / 60
        
        # Create summary
        summary = {
            'experiment': 'FL + GradInversion',
            'device': self.device,
            'start_time': experiment_start.isoformat(),
            'end_time': experiment_end.isoformat(),
            'duration_minutes': round(duration, 2),
            'baseline_results': baseline_results,
            'attack_results': attack_results
        }
        
        with open('results/experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\n{'█'*80}")
        print(f"Experiment completed: {experiment_end.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration:.2f} minutes")
        print(f"{'█'*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning + GradInversion Privacy Attack Experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fedavg_baseline.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "attack", "full"],
        default="full",
        help="Experiment mode"
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create runner
    runner = FLExperimentRunner(args.config)
    
    # Run experiments
    if args.mode == "baseline":
        runner.run_baseline_fedavg()
    elif args.mode == "attack":
        runner.run_gradinversion_attack()
    else:  # full
        runner.run_full_experiment()


if __name__ == "__main__":
    main()