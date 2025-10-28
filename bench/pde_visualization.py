"""
Visualization Suite for PDE Discovery

Generates comprehensive visualizations for the Visual Critic agent:
1. Temporal snapshots: observed vs predicted
2. Difference maps: spatial error distribution
3. Temporal evolution: spatially-averaged quantities
4. Gradient field comparison
5. Fourier spectrum analysis
6. Conservation verification
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import io
from PIL import Image


class PDEVisualizer:
    """Visualization suite for spatiotemporal PDE solutions"""

    def __init__(self, figsize: Tuple[int, int] = (16, 12), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        plt.rcParams['font.size'] = 10

    def create_comprehensive_plot(self, observed: np.ndarray, predicted: np.ndarray,
                                 equation_str: str = "",
                                 score: float = None,
                                 save_path: Optional[str] = None) -> Image.Image:
        """
        Create comprehensive multi-panel visualization

        Args:
            observed: Observed field (H, W, T)
            predicted: Predicted field (H, W, T)
            equation_str: PDE equation string
            score: Numerical score
            save_path: Optional path to save figure

        Returns:
            PIL Image of the figure
        """
        H, W, T = observed.shape

        # Create figure with custom layout
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Title
        title = f"PDE Evaluation"
        if equation_str:
            title += f": {equation_str[:80]}"
        if score is not None:
            title += f" | Score: {score:.3f}"
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # 1. Temporal snapshots (3 timepoints)
        times = [0, T//2, T-1]
        time_labels = ['Early', 'Mid', 'Late']

        for i, (t, label) in enumerate(zip(times, time_labels)):
            # Observed
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(observed[:, :, t], cmap='viridis', aspect='auto')
            ax.set_title(f'Observed ({label})')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

            # Predicted
            ax = fig.add_subplot(gs[1, i])
            im = ax.imshow(predicted[:, :, t], cmap='viridis', aspect='auto',
                          vmin=observed[:, :, t].min(), vmax=observed[:, :, t].max())
            ax.set_title(f'Predicted ({label})')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # 2. Difference maps
        for i, (t, label) in enumerate(zip(times, time_labels)):
            ax = fig.add_subplot(gs[0, 3])  # Only show last timepoint
            if i == len(times) - 1:
                diff = predicted[:, :, t] - observed[:, :, t]
                max_abs = np.max(np.abs(diff))
                im = ax.imshow(diff, cmap='RdBu_r', aspect='auto',
                              vmin=-max_abs, vmax=max_abs)
                ax.set_title(f'Error Map (Late)')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, label='Pred - Obs')

        # 3. Temporal evolution (spatial averages)
        ax = fig.add_subplot(gs[1, 3])
        time_axis = np.arange(T)
        obs_mean = np.mean(observed, axis=(0, 1))
        pred_mean = np.mean(predicted, axis=(0, 1))
        ax.plot(time_axis, obs_mean, 'b-', label='Observed', linewidth=2)
        ax.plot(time_axis, pred_mean, 'r--', label='Predicted', linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Spatial Mean Density')
        ax.set_title('Temporal Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Gradient field comparison (mid timepoint)
        t_mid = T // 2
        ax1 = fig.add_subplot(gs[2, 0])
        ax2 = fig.add_subplot(gs[2, 1])

        # Subsample for quiver plot
        stride = max(H // 20, 1)
        X, Y = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))

        # Compute gradients
        obs_grad_y, obs_grad_x = np.gradient(observed[:, :, t_mid])
        pred_grad_y, pred_grad_x = np.gradient(predicted[:, :, t_mid])

        # Observed gradient
        ax1.quiver(X, Y,
                  obs_grad_x[::stride, ::stride],
                  -obs_grad_y[::stride, ::stride],  # Flip y for image coordinates
                  alpha=0.7)
        ax1.imshow(observed[:, :, t_mid], cmap='gray', alpha=0.3, aspect='auto')
        ax1.set_title('Observed Gradient (Mid)')
        ax1.axis('off')

        # Predicted gradient
        ax2.quiver(X, Y,
                  pred_grad_x[::stride, ::stride],
                  -pred_grad_y[::stride, ::stride],
                  alpha=0.7, color='red')
        ax2.imshow(predicted[:, :, t_mid], cmap='gray', alpha=0.3, aspect='auto')
        ax2.set_title('Predicted Gradient (Mid)')
        ax2.axis('off')

        # 5. Fourier spectrum analysis
        ax1 = fig.add_subplot(gs[2, 2])
        ax2 = fig.add_subplot(gs[2, 3])

        # Compute 2D FFT for last timepoint
        obs_fft = np.abs(np.fft.fftshift(np.fft.fft2(observed[:, :, -1])))
        pred_fft = np.abs(np.fft.fftshift(np.fft.fft2(predicted[:, :, -1])))

        # Log scale for better visualization
        obs_fft_log = np.log10(obs_fft + 1e-10)
        pred_fft_log = np.log10(pred_fft + 1e-10)

        im1 = ax1.imshow(obs_fft_log, cmap='hot', aspect='auto')
        ax1.set_title('Observed Spectrum (Late)')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, label='log10(|FFT|)')

        im2 = ax2.imshow(pred_fft_log, cmap='hot', aspect='auto')
        ax2.set_title('Predicted Spectrum (Late)')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, label='log10(|FFT|)')

        # 6. Conservation verification
        ax = fig.add_subplot(gs[3, :2])
        time_axis = np.arange(T)
        obs_total_mass = np.sum(observed, axis=(0, 1))
        pred_total_mass = np.sum(predicted, axis=(0, 1))

        ax.plot(time_axis, obs_total_mass, 'b-', label='Observed', linewidth=2)
        ax.plot(time_axis, pred_total_mass, 'r--', label='Predicted', linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Total Mass (Σ g)')
        ax.set_title('Mass Conservation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 7. Error metrics over time
        ax = fig.add_subplot(gs[3, 2:])
        mse_over_time = np.mean((predicted - observed)**2, axis=(0, 1))
        ax.plot(time_axis, mse_over_time, 'k-', linewidth=2)
        ax.fill_between(time_axis, 0, mse_over_time, alpha=0.3, color='red')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('MSE')
        ax.set_title('Prediction Error Over Time')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Save or convert to image
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
        buf.seek(0)
        img = Image.open(buf)

        plt.close(fig)

        return img

    def create_simple_comparison(self, observed: np.ndarray, predicted: np.ndarray,
                                timepoint: int = -1) -> Image.Image:
        """
        Create simple side-by-side comparison at a single timepoint

        Args:
            observed: Observed field (H, W, T)
            predicted: Predicted field (H, W, T)
            timepoint: Time index to visualize

        Returns:
            PIL Image
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=self.dpi)

        # Observed
        im1 = axes[0].imshow(observed[:, :, timepoint], cmap='viridis', aspect='auto')
        axes[0].set_title('Observed')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])

        # Predicted
        im2 = axes[1].imshow(predicted[:, :, timepoint], cmap='viridis', aspect='auto',
                            vmin=observed[:, :, timepoint].min(),
                            vmax=observed[:, :, timepoint].max())
        axes[1].set_title('Predicted')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])

        # Difference
        diff = predicted[:, :, timepoint] - observed[:, :, timepoint]
        max_abs = np.max(np.abs(diff))
        im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto',
                            vmin=-max_abs, vmax=max_abs)
        axes[2].set_title('Error (Pred - Obs)')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
        buf.seek(0)
        img = Image.open(buf)

        plt.close(fig)

        return img

    def create_animation_frames(self, observed: np.ndarray, predicted: np.ndarray,
                               output_dir: str, num_frames: int = 50) -> List[str]:
        """
        Create animation frames showing temporal evolution

        Args:
            observed: Observed field (H, W, T)
            predicted: Predicted field (H, W, T)
            output_dir: Directory to save frames
            num_frames: Number of frames to generate

        Returns:
            List of frame file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        H, W, T = observed.shape
        frame_indices = np.linspace(0, T-1, num_frames, dtype=int)
        frame_paths = []

        vmin = min(observed.min(), predicted.min())
        vmax = max(observed.max(), predicted.max())

        for i, t in enumerate(frame_indices):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=self.dpi)

            axes[0].imshow(observed[:, :, t], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
            axes[0].set_title(f'Observed (t={t})')
            axes[0].axis('off')

            axes[1].imshow(predicted[:, :, t], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
            axes[1].set_title(f'Predicted (t={t})')
            axes[1].axis('off')

            frame_path = output_path / f'frame_{i:04d}.png'
            plt.savefig(frame_path, bbox_inches='tight', dpi=self.dpi)
            plt.close(fig)

            frame_paths.append(str(frame_path))

        return frame_paths

    def create_critique_visualization(self, observed: np.ndarray, predicted: np.ndarray,
                                     equation_str: str, metrics: Dict,
                                     save_path: Optional[str] = None) -> Image.Image:
        """
        Create focused visualization for Visual Critic analysis

        Args:
            observed: Observed field (H, W, T)
            predicted: Predicted field (H, W, T)
            equation_str: Equation string
            metrics: Dictionary of computed metrics
            save_path: Optional save path

        Returns:
            PIL Image
        """
        H, W, T = observed.shape

        fig = plt.figure(figsize=(14, 10), dpi=self.dpi)
        gs = GridSpec(3, 3, figure=fig, hspace=0.25, wspace=0.3)

        # Title with metrics
        title = f"Visual Critique: {equation_str[:60]}\n"
        title += f"MSE: {metrics.get('mse', 0):.6f} | R²: {metrics.get('r2', 0):.4f} | NMSE: {metrics.get('nmse', 0):.4f}"
        fig.suptitle(title, fontsize=12, fontweight='bold')

        # Row 1: Snapshots at 3 timepoints
        times = [0, T//2, T-1]
        labels = ['t=0', f't={T//2}', f't={T-1}']

        for i, (t, label) in enumerate(zip(times, labels)):
            ax = fig.add_subplot(gs[0, i])

            # Stack observed and predicted side by side
            combined = np.hstack([observed[:, :, t], predicted[:, :, t]])
            im = ax.imshow(combined, cmap='viridis', aspect='auto')
            ax.set_title(f'{label} | Obs (L) vs Pred (R)')
            ax.axis('off')

            # Add vertical line separator
            ax.axvline(x=W-0.5, color='white', linewidth=2, linestyle='--')

        # Row 2: Error analysis
        ax1 = fig.add_subplot(gs[1, 0])
        diff_late = predicted[:, :, -1] - observed[:, :, -1]
        max_abs = np.max(np.abs(diff_late))
        im = ax1.imshow(diff_late, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs, aspect='auto')
        ax1.set_title('Error Map (Late)')
        ax1.axis('off')
        plt.colorbar(im, ax=ax1, fraction=0.046)

        ax2 = fig.add_subplot(gs[1, 1])
        mse_time = np.mean((predicted - observed)**2, axis=(0, 1))
        ax2.plot(mse_time, 'r-', linewidth=2)
        ax2.fill_between(range(T), 0, mse_time, alpha=0.3, color='red')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('MSE')
        ax2.set_title('Error Evolution')
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 2])
        # Error histogram
        all_errors = (predicted - observed).flatten()
        ax3.hist(all_errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect')
        ax3.set_xlabel('Error (Pred - Obs)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Row 3: Physical properties
        ax1 = fig.add_subplot(gs[2, 0])
        obs_mean = np.mean(observed, axis=(0, 1))
        pred_mean = np.mean(predicted, axis=(0, 1))
        ax1.plot(obs_mean, 'b-', label='Observed', linewidth=2)
        ax1.plot(pred_mean, 'r--', label='Predicted', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Mean Density')
        ax1.set_title('Spatial Average')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[2, 1])
        obs_std = np.std(observed, axis=(0, 1))
        pred_std = np.std(predicted, axis=(0, 1))
        ax2.plot(obs_std, 'b-', label='Observed', linewidth=2)
        ax2.plot(pred_std, 'r--', label='Predicted', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Spatial Std Dev')
        ax2.set_title('Spread Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[2, 2])
        obs_mass = np.sum(observed, axis=(0, 1))
        pred_mass = np.sum(predicted, axis=(0, 1))
        ax3.plot(obs_mass, 'b-', label='Observed', linewidth=2)
        ax3.plot(pred_mass, 'r--', label='Predicted', linewidth=2)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Total Mass')
        ax3.set_title('Mass Conservation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
        buf.seek(0)
        img = Image.open(buf)

        plt.close(fig)

        return img

    def create_rollout_grid_with_stats(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        times: Optional[List[int]] = None,
        save_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Create a 9-panel figure with 6 rollout visualizations and 3 statistical plots.

        - Panels 1-6: Obs (left) vs Pred (right) for six timepoints
        - Panel 7: MSE over time
        - Panel 8: Total mass over time (Obs vs Pred)
        - Panel 9: Spatial mean over time (Obs vs Pred)

        Args:
            observed: (H, W, T)
            predicted: (H, W, T)
            times: list of 6 time indices to visualize; if None, uses 6 evenly spaced
            save_path: optional output path
        Returns:
            PIL Image of the figure
        """
        H, W, T = observed.shape
        assert predicted.shape == observed.shape, "Observed and predicted must share shape"

        if times is None:
            # Evenly spaced including first and last; ensure 6 unique indices
            times = np.linspace(0, T-1, 6, dtype=int).tolist()
            # Guarantee uniqueness and sorted
            times = sorted(list(dict.fromkeys(times)))
            # If fewer than 6 due to very small T, pad with last
            while len(times) < 6:
                times.append(T-1)

        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Panels 1-6: rollout frames (Obs vs Pred combined horizontally)
        vmin = np.min(observed)
        vmax = np.max(observed)
        for i, t in enumerate(times[:6]):
            row, col = divmod(i, 3)
            ax = fig.add_subplot(gs[row, col])
            t = int(np.clip(t, 0, T-1))
            combined = np.hstack([observed[:, :, t], predicted[:, :, t]])
            im = ax.imshow(combined, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title(f'Rollout t={t} | Obs (L) vs Pred (R)')
            ax.axis('off')
            # Separator line between Obs and Pred halves
            ax.axvline(x=W-0.5, color='white', linewidth=2, linestyle='--')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Panel 7: MSE over time
        ax7 = fig.add_subplot(gs[2, 0])
        mse_over_time = np.mean((predicted - observed) ** 2, axis=(0, 1))
        ax7.plot(mse_over_time, 'k-', linewidth=2, label='MSE')
        ax7.fill_between(np.arange(T), 0, mse_over_time, alpha=0.25, color='red')
        ax7.set_title('MSE Over Time')
        ax7.set_xlabel('Time')
        ax7.set_ylabel('MSE')
        ax7.grid(True, alpha=0.3)
        ax7.set_yscale('log')

        # Panel 8: Total mass over time
        ax8 = fig.add_subplot(gs[2, 1])
        obs_mass = np.sum(observed, axis=(0, 1))
        pred_mass = np.sum(predicted, axis=(0, 1))
        ax8.plot(obs_mass, 'b-', linewidth=2, label='Observed')
        ax8.plot(pred_mass, 'r--', linewidth=2, label='Predicted')
        ax8.set_title('Total Mass Over Time')
        ax8.set_xlabel('Time')
        ax8.set_ylabel('Σ g')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # Panel 9: Spatial mean over time
        ax9 = fig.add_subplot(gs[2, 2])
        obs_mean = np.mean(observed, axis=(0, 1))
        pred_mean = np.mean(predicted, axis=(0, 1))
        ax9.plot(obs_mean, 'b-', linewidth=2, label='Observed')
        ax9.plot(pred_mean, 'r--', linewidth=2, label='Predicted')
        ax9.set_title('Spatial Mean Over Time')
        ax9.set_xlabel('Time')
        ax9.set_ylabel('Mean g')
        ax9.legend()
        ax9.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img
