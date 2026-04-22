"""
Utility functions for evaluation, visualization, and metrics
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
from sklearn.metrics import auc
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class MetricsComputer:
    """Compute Category 1 (Privacy) metrics"""
    
    @staticmethod
    def reconstruction_mse(
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> float:
        """Mean Squared Error between images"""
        return torch.mean((original - reconstructed) ** 2).item()
    
    @staticmethod
    def reconstruction_psnr(
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        max_pixel: float = 1.0
    ) -> float:
        """Peak Signal-to-Noise Ratio"""
        mse = torch.mean((original - reconstructed) ** 2)
        if mse < 1e-10:
            return 100.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()
    
    @staticmethod
    def reconstruction_ssim(
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        window_size: int = 11
    ) -> float:
        """Structural Similarity Index (simplified)"""
        # Simplified SSIM without windowing
        mean_orig = torch.mean(original)
        mean_recon = torch.mean(reconstructed)
        
        var_orig = torch.var(original)
        var_recon = torch.var(reconstructed)
        cov = torch.mean((original - mean_orig) * (reconstructed - mean_recon))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mean_orig * mean_recon + c1) * (2 * cov + c2)) / (
            (mean_orig ** 2 + mean_recon ** 2 + c1) * (var_orig + var_recon + c2)
        )
        
        return ssim.item()
    
    @staticmethod
    def attack_success_rate(
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        mse_threshold: float = 0.02
    ) -> float:
        """Success if MSE < threshold (0-1)"""
        mse = torch.mean((original - reconstructed) ** 2).item()
        return 1.0 if mse < mse_threshold else 0.0
    
    @staticmethod
    def gradient_matching_error(
        target_grads: Dict[str, torch.Tensor],
        computed_grads: Dict[str, torch.Tensor]
    ) -> float:
        """L2 norm of gradient difference"""
        total_error = 0.0
        for name, target_grad in target_grads.items():
            if name in computed_grads:
                error = torch.norm(computed_grads[name] - target_grad) ** 2
                total_error += error.item()
        return np.sqrt(total_error)


class ResultsVisualizer:
    """Visualization utilities for FL experiments"""
    
    @staticmethod
    def plot_accuracy_vs_rounds(
        results: Dict[str, List[float]],
        save_path: str = "results/plots/accuracy_vs_rounds.png",
        title: str = "Global Accuracy vs Communication Rounds"
    ):
        """Plot global accuracy over communication rounds"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(12, 7), dpi=300)
        
        for method, accuracies in results.items():
            rounds = range(1, len(accuracies) + 1)
            plt.plot(rounds, accuracies, marker='o', linewidth=2.5, 
                    label=method, markersize=5)
        
        plt.xlabel('Communication Rounds', fontsize=13, fontweight='bold')
        plt.ylabel('Global Test Accuracy (%)', fontsize=13, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.legend(fontsize=11, loc='lower right')
        plt.grid(True, alpha=0.4)
        plt.ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {save_path}")
    
    @staticmethod
    def plot_loss_vs_rounds(
        results: Dict[str, List[float]],
        save_path: str = "results/plots/loss_vs_rounds.png",
        title: str = "Global Loss vs Communication Rounds"
    ):
        """Plot global loss over rounds"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(12, 7), dpi=300)
        
        for method, losses in results.items():
            rounds = range(1, len(losses) + 1)
            plt.plot(rounds, losses, marker='s', linewidth=2.5,
                    label=method, markersize=5)
        
        plt.xlabel('Communication Rounds', fontsize=13, fontweight='bold')
        plt.ylabel('Global Test Loss', fontsize=13, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.legend(fontsize=11, loc='upper right')
        plt.grid(True, alpha=0.4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {save_path}")
    
    @staticmethod
    def plot_attack_success_rate(
        batch_sizes: List[int],
        success_rates: List[float],
        save_path: str = "results/plots/attack_success_rate.png",
        title: str = "GradInversion Attack Success Rate vs Batch Size"
    ):
        """Plot attack success rate vs batch size"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(11, 7), dpi=300)
        
        colors = ['#2ecc71' if sr > 70 else '#e74c3c' for sr in success_rates]
        bars = plt.bar(range(len(batch_sizes)), success_rates, 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        plt.xticks(range(len(batch_sizes)), batch_sizes, fontsize=12, fontweight='bold')
        plt.xlabel('Batch Size', fontsize=13, fontweight='bold')
        plt.ylabel('Attack Success Rate (%)', fontsize=13, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.ylim([0, 105])
        
        # Add value labels
        for i, (bar, sr) in enumerate(zip(bars, success_rates)):
            plt.text(i, sr + 2, f'{sr:.1f}%', ha='center', fontsize=11, fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {save_path}")
    
    @staticmethod
    def plot_reconstruction_comparison(
        original_batch: torch.Tensor,
        reconstructed_batch: torch.Tensor,
        save_path: str = "results/plots/reconstruction_comparison.png",
        num_images: int = 8
    ):
        """Visualize original vs reconstructed images"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        num_show = min(num_images, len(original_batch))
        fig, axes = plt.subplots(2, num_show, figsize=(16, 4), dpi=300)
        
        orig_np = original_batch.cpu().numpy()
        recon_np = reconstructed_batch.cpu().numpy()
        
        # Denormalize CIFAR-10
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
        
        for i in range(num_show):
            # Original
            orig_img = np.transpose(orig_np[i], (1, 2, 0))
            orig_img = np.clip(orig_img * std.reshape(3) + mean.reshape(3), 0, 1)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'Original {i}', fontsize=10, fontweight='bold')
            axes[0, i].axis('off')
            
            # Reconstructed
            recon_img = np.transpose(recon_np[i], (1, 2, 0))
            recon_img = np.clip(recon_img * std.reshape(3) + mean.reshape(3), 0, 1)
            axes[1, i].imshow(recon_img)
            axes[1, i].set_title(f'Reconstructed {i}', fontsize=10, fontweight='bold')
            axes[1, i].axis('off')
        
        plt.suptitle('Image Reconstruction: Original vs GradInversion', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {save_path}")
    
    @staticmethod
    def plot_reconstruction_metrics(
        metrics: Dict[str, List[float]],
        save_path: str = "results/plots/reconstruction_metrics.png"
    ):
        """Plot reconstruction quality metrics"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
        
        # MSE
        if 'mse' in metrics:
            axes[0].plot(metrics['mse'], marker='o', linewidth=2, color='#e74c3c')
            axes[0].set_ylabel('MSE', fontsize=12, fontweight='bold')
            axes[0].set_title('Reconstruction MSE (Lower is Better)', fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
        
        # PSNR
        if 'psnr' in metrics:
            axes[1].plot(metrics['psnr'], marker='s', linewidth=2, color='#3498db')
            axes[1].set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
            axes[1].set_title('Peak Signal-to-Noise Ratio (Higher is Better)', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
        
        # SSIM
        if 'ssim' in metrics:
            axes[2].plot(metrics['ssim'], marker='^', linewidth=2, color='#2ecc71')
            axes[2].set_ylabel('SSIM', fontsize=12, fontweight='bold')
            axes[2].set_title('Structural Similarity (Higher is Better)', fontsize=12, fontweight='bold')
            axes[2].set_ylim([0, 1])
            axes[2].grid(True, alpha=0.3)
        
        for ax in axes:
            ax.set_xlabel('Batch Index', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {save_path}")
    
    @staticmethod
    def plot_iid_vs_noniid(
        iid_acc: List[float],
        noniid_acc: List[float],
        save_path: str = "results/plots/iid_vs_noniid.png"
    ):
        """Compare IID vs Non-IID convergence"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(12, 7), dpi=300)
        
        rounds = range(1, len(iid_acc) + 1)
        plt.plot(rounds, iid_acc, marker='o', linewidth=2.5, label='IID (α=1.0)',
                color='#2ecc71', markersize=5)
        plt.plot(rounds, noniid_acc, marker='s', linewidth=2.5, label='Non-IID (α=0.1)',
                color='#e74c3c', markersize=5)
        
        plt.xlabel('Communication Rounds', fontsize=13, fontweight='bold')
        plt.ylabel('Global Test Accuracy (%)', fontsize=13, fontweight='bold')
        plt.title('IID vs Non-IID Data Distribution', fontsize=14, fontweight='bold', pad=15)
        plt.legend(fontsize=12, loc='lower right')
        plt.grid(True, alpha=0.4)
        plt.ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {save_path}")


class ResultsExporter:
    """Export results to various formats"""
    
    @staticmethod
    def export_to_csv(
        results: List[Dict],
        save_path: str = "results/metrics.csv"
    ):
        """Export results to CSV"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(results)
        df.to_csv(save_path, index=False)
        
        print(f"✓ Saved CSV: {save_path}")
        print("\nResults Summary:")
        print(df.to_string(index=False))
        return df
    
    @staticmethod
    def export_to_json(
        results: Dict,
        save_path: str = "results/results.json"
    ):
        """Export results to JSON"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"✓ Saved JSON: {save_path}")
    
    @staticmethod
    def create_results_table(
        experiments: List[Dict],
        save_path: str = "results/results_table.csv"
    ):
        """Create formatted results table"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(experiments)
        
        # Format numeric columns
        numeric_cols = ['Test Acc (%)', 'Test Loss', 'Comm. Cost (MB)', 
                       'Attack Success (%)', 'Recon. MSE']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        
        df.to_csv(save_path, index=False)
        print(f"✓ Saved results table: {save_path}")
        print("\n" + "="*80)
        print(df.to_string(index=False))
        print("="*80)


def create_experiment_summary(
    experiment_name: str,
    dataset: str,
    num_clients: int,
    num_rounds: int,
    batch_size: int,
    test_accuracy: float,
    test_loss: float,
    comm_cost_mb: float,
    attack_success_rate: float,
    recon_mse: float,
    save_path: str = "results/summary.json"
) -> Dict:
    """Create experiment summary"""
    
    summary = {
        "experiment": experiment_name,
        "dataset": dataset,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "batch_size": batch_size,
        "test_accuracy_percent": round(test_accuracy, 4),
        "test_loss": round(test_loss, 4),
        "communication_cost_mb": round(comm_cost_mb, 2),
        "attack_success_rate_percent": round(attack_success_rate, 4),
        "reconstruction_mse": round(recon_mse, 6)
    }
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    return summary