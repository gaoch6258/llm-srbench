#!/usr/bin/env python3
"""
Create Complex PDE Test Case - Version 2 (More Dynamic)

Generates a challenging test case with better temporal dynamics:
Complex: ∂g/∂t = α·Δg - β·∇·(g∇(ln S)) + γ·g(1-g/K)

With stronger parameters for visible dynamics.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from bench.pde_solver import PDESolver, PDEConfig
from bench.pde_datamodule import ChemotaxisDataModule, ChemotaxisProblem


class ComplexPDESolver(PDESolver):
    """Extended solver with logistic growth term"""

    def solve_complex_chemotaxis(self, g_init: np.ndarray, S: np.ndarray,
                                 alpha: float, beta: float, gamma: float, K: float,
                                 num_steps: int = None) -> np.ndarray:
        """
        Solve: ∂g/∂t = α·Δg - β·∇·(g∇(ln S)) + γ·g(1-g/K)
        """
        num_steps = num_steps or self.config.max_iterations
        H, W = g_init.shape
        g_history = np.zeros((H, W, num_steps))
        g_history[:, :, 0] = g_init.copy()

        g_current = g_init.copy()

        for t in range(1, num_steps):
            if S.ndim == 3:
                S_current = S[:, :, min(t, S.shape[2]-1)]
            else:
                S_current = S

            # Diffusion
            diffusion = alpha * self.laplacian_2d(g_current)

            # Chemotaxis
            chemotaxis = -beta * self.chemotaxis_term(g_current, S_current)

            # Logistic growth
            logistic = gamma * g_current * (1 - g_current / K)

            # Update
            g_next = g_current + self.config.dt * (diffusion + chemotaxis + logistic)
            g_next = np.maximum(g_next, 0)
            g_next = np.minimum(g_next, K * 1.2)

            g_history[:, :, t] = g_next
            g_current = g_next

        return g_history


def create_complex_test_case_v2(output_dir: str = "./logs/pde_discovery_complex"):
    """Create complex test case with stronger dynamics"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CREATING COMPLEX PDE TEST CASE V2 (More Dynamic)")
    print("="*70)

    # Configuration - Stronger parameters for visible dynamics
    H, W = 128, 128
    T = 200

    # Stronger parameters
    alpha_true = 0.5    # Higher diffusion
    beta_true = 1.5     # Stronger chemotaxis
    gamma_true = 0.15   # Faster growth
    K_true = 3.0        # Higher capacity

    print(f"\n1. True PDE Parameters (Stronger for visibility):")
    print(f"   α (diffusion):  {alpha_true}")
    print(f"   β (chemotaxis): {beta_true}")
    print(f"   γ (growth):     {gamma_true}")
    print(f"   K (capacity):   {K_true}")

    # Smaller dt for stability with stronger parameters
    solver = ComplexPDESolver(PDEConfig(dx=1.0, dy=1.0, dt=0.005, max_iterations=T))

    # Spatial grid
    x = np.linspace(0, H-1, H)
    y = np.linspace(0, W-1, W)
    X, Y = np.meshgrid(x, y)

    # Initial: Three small populations
    print(f"\n2. Creating initial condition (3 small cell populations)...")
    g_init = (
        0.3 * np.exp(-((X - H/4)**2 + (Y - W/4)**2) / (0.02 * H**2)) +
        0.4 * np.exp(-((X - H/2)**2 + (Y - 3*W/4)**2) / (0.02 * H**2)) +
        0.25 * np.exp(-((X - 3*H/4)**2 + (Y - W/2)**2) / (0.015 * H**2))
    )

    # Stronger chemoattractant gradient
    print(f"3. Creating strong chemoattractant gradient...")
    S = (
        3.0 * np.exp(0.03 * (X - H/2) + 0.025 * (Y - W/2)) +  # Strong directional gradient
        5.0 * np.exp(-((X - 3*H/4)**2 + (Y - 3*W/4)**2) / (0.08 * H**2))  # Strong attractant source
    )
    S = S / S.max()

    # Solve
    print(f"\n4. Solving complex PDE...")
    print(f"   Grid: {H}×{W}, Timepoints: {T}, dt: {solver.config.dt}")

    g_observed = solver.solve_complex_chemotaxis(
        g_init, S, alpha_true, beta_true, gamma_true, K_true, num_steps=T
    )

    # Statistics
    print(f"\n5. Data statistics:")
    print(f"   Initial mass: {g_init.sum():.4f}")
    print(f"   Final mass:   {g_observed[:, :, -1].sum():.4f}")
    print(f"   Mass change:  {(g_observed[:, :, -1].sum() - g_init.sum()):.4f} ({(g_observed[:, :, -1].sum() / g_init.sum() - 1) * 100:.1f}%)")
    print(f"   Max density:  {g_observed.max():.4f}")
    print(f"   Initial max:  {g_init.max():.4f}")
    print(f"   Growth ratio: {g_observed.max() / g_init.max():.2f}x")

    # Visualization
    print(f"\n6. Creating visualization...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Row 1: Cell density evolution
    times = [0, T//4, T//2, T-1]
    for i, t in enumerate(times):
        ax = axes[0, i]
        im = ax.imshow(g_observed[:, :, t], cmap='viridis', vmin=0, vmax=g_observed.max(), aspect='auto')
        ax.set_title(f'Cell Density t={t}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2: Difference from initial
    for i, t in enumerate(times[1:], 1):
        ax = axes[1, i]
        diff = g_observed[:, :, t] - g_init
        max_abs = max(abs(diff.min()), abs(diff.max()))
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs, aspect='auto')
        ax.set_title(f'Change from t=0 (t={t})')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    ax.imshow(S, cmap='hot', aspect='auto')
    ax.set_title('Chemoattractant Field S')
    ax.axis('off')

    # Row 3: Analysis plots
    ax = axes[2, 0]
    total_mass = np.sum(g_observed, axis=(0, 1))
    ax.plot(total_mass, 'b-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Mass')
    ax.set_title(f'Mass Evolution (+{(total_mass[-1]/total_mass[0]-1)*100:.1f}%)')
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    mean_density = np.mean(g_observed, axis=(0, 1))
    ax.plot(mean_density, 'g-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean Density')
    ax.set_title('Spatial Average')
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    max_density = np.max(g_observed, axis=(0, 1))
    ax.plot(max_density, 'r-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Max Density')
    ax.set_title('Peak Density')
    ax.grid(True, alpha=0.3)

    ax = axes[2, 3]
    std_density = np.std(g_observed, axis=(0, 1))
    ax.plot(std_density, 'm-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Std Dev')
    ax.set_title('Spatial Heterogeneity')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    viz_path = output_path / "complex_pde_overview_v2.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved to: {viz_path}")

    # Save dataset
    gt_equation = "∂g/∂t = α·Δg - β·∇·(g∇(ln S)) + γ·g(1-g/K)"

    problem = ChemotaxisProblem(
        g_init=g_init,
        S=S,
        g_observed=g_observed,
        metadata={
            'alpha_true': alpha_true,
            'beta_true': beta_true,
            'gamma_true': gamma_true,
            'K_true': K_true,
            'H': H, 'W': W, 'T': T,
            'dx': 1.0, 'dy': 1.0, 'dt': 0.005,
            'reference_pde': gt_equation,
            'description': 'Complex chemotaxis with logistic growth - Dynamic version',
            'difficulty': 'hard',
            'version': 2
        },
        gt_equation=gt_equation
    )

    dm = ChemotaxisDataModule(data_source="synthetic")
    dm.problems = {'complex_v2_001': problem}

    dataset_path = output_path / "complex_chemotaxis_v2.hdf5"
    dm.save_hdf5(str(dataset_path))
    print(f"\n7. Dataset saved to: {dataset_path}")

    numpy_path = output_path / "complex_chemotaxis_v2.npz"
    np.savez(
        numpy_path,
        g_init=g_init,
        S=S,
        g_observed=g_observed,
        alpha_true=alpha_true,
        beta_true=beta_true,
        gamma_true=gamma_true,
        K_true=K_true,
        gt_equation=gt_equation
    )
    print(f"   Also saved: {numpy_path}")

    print(f"\n" + "="*70)
    print("COMPLEX TEST CASE V2 CREATED!")
    print("="*70)
    print(f"\nGround Truth: {gt_equation}")
    print(f"\nTrue Parameters:")
    print(f"  α = {alpha_true} (diffusion)")
    print(f"  β = {beta_true} (chemotaxis)")
    print(f"  γ = {gamma_true} (growth)")
    print(f"  K = {K_true} (capacity)")
    print(f"\nDynamics: {(total_mass[-1]/total_mass[0]-1)*100:.1f}% mass increase")
    print(f"Peak growth: {max_density.max()/g_init.max():.2f}x initial maximum")

    return str(dataset_path)


if __name__ == "__main__":
    create_complex_test_case_v2()
