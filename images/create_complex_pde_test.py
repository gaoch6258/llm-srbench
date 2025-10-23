#!/usr/bin/env python3
"""
Create Complex PDE Test Case

Generates a challenging test case with a modified chemotaxis PDE:
Reference: ∂g/∂t = α·Δg - ∇·(g∇(ln S))
Complex:   ∂g/∂t = α·Δg - β·∇·(g∇(ln S)) + γ·g(1-g/K)

This adds:
- β: weighted chemotaxis term (allows testing parameter discovery)
- γ·g(1-g/K): logistic growth/saturation term (density-dependent)
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from bench.pde_solver import PDESolver, PDEConfig
from bench.pde_datamodule import ChemotaxisDataModule, ChemotaxisProblem


class ComplexPDESolver(PDESolver):
    """Extended solver with additional terms"""

    def solve_complex_chemotaxis(self, g_init: np.ndarray, S: np.ndarray,
                                 alpha: float, beta: float, gamma: float, K: float,
                                 num_steps: int = None) -> np.ndarray:
        """
        Solve complex chemotaxis PDE with logistic growth:
        ∂g/∂t = α·Δg - β·∇·(g∇(ln S)) + γ·g(1-g/K)

        Args:
            g_init: Initial condition
            S: Chemoattractant field
            alpha: Diffusion coefficient
            beta: Chemotaxis coefficient
            gamma: Growth rate
            K: Carrying capacity
            num_steps: Number of time steps

        Returns:
            Evolution (H, W, T)
        """
        if self.config.stability_check:
            # Check diffusion stability
            if not self.check_cfl_condition(alpha):
                print(f"Warning: CFL violated for diffusion with α={alpha}")

        num_steps = num_steps or self.config.max_iterations
        H, W = g_init.shape
        g_history = np.zeros((H, W, num_steps))
        g_history[:, :, 0] = g_init.copy()

        g_current = g_init.copy()

        for t in range(1, num_steps):
            # Chemoattractant at current time
            if S.ndim == 3:
                S_current = S[:, :, min(t, S.shape[2]-1)]
            else:
                S_current = S

            # Term 1: Diffusion α·Δg
            diffusion = alpha * self.laplacian_2d(g_current)

            # Term 2: Chemotaxis -β·∇·(g∇(ln S))
            chemotaxis = -beta * self.chemotaxis_term(g_current, S_current)

            # Term 3: Logistic growth γ·g(1-g/K)
            logistic = gamma * g_current * (1 - g_current / K)

            # Update
            g_next = g_current + self.config.dt * (diffusion + chemotaxis + logistic)

            # Ensure non-negativity and don't exceed carrying capacity
            g_next = np.maximum(g_next, 0)
            g_next = np.minimum(g_next, K * 1.1)  # Allow slight overshoot

            g_history[:, :, t] = g_next
            g_current = g_next

        return g_history


def create_complex_test_case(output_dir: str = "./logs/pde_discovery_complex"):
    """
    Create complex test case with modified PDE

    Returns:
        Path to saved dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CREATING COMPLEX PDE TEST CASE")
    print("="*70)

    # Configuration
    H, W = 128, 128  # Reasonable size for testing
    T = 150          # More timepoints to capture dynamics

    # True parameters (hidden from discovery system)
    alpha_true = 0.3    # Diffusion coefficient
    beta_true = 0.8     # Chemotaxis coefficient
    gamma_true = 0.05   # Growth rate
    K_true = 2.0        # Carrying capacity

    print(f"\n1. True PDE Parameters:")
    print(f"   α (diffusion):  {alpha_true}")
    print(f"   β (chemotaxis): {beta_true}")
    print(f"   γ (growth):     {gamma_true}")
    print(f"   K (capacity):   {K_true}")

    # Setup
    solver = ComplexPDESolver(PDEConfig(dx=1.0, dy=1.0, dt=0.01, max_iterations=T))

    # Create spatial grid
    x = np.linspace(0, H-1, H)
    y = np.linspace(0, W-1, W)
    X, Y = np.meshgrid(x, y)

    # Initial condition: Two Gaussian blobs
    print(f"\n2. Creating initial condition (2 cell populations)...")
    g_init = (
        0.5 * np.exp(-((X - H/3)**2 + (Y - W/3)**2) / (0.05 * H**2)) +
        0.3 * np.exp(-((X - 2*H/3)**2 + (Y - 2*W/3)**2) / (0.03 * H**2))
    )

    # Chemoattractant field: More complex gradient with multiple sources
    print(f"3. Creating chemoattractant field (3 sources)...")
    S = (
        np.exp(0.02 * X + 0.01 * Y) +  # Global gradient
        2.0 * np.exp(-((X - H/2)**2 + (Y - W/4)**2) / (0.1 * H**2)) +  # Source 1
        1.5 * np.exp(-((X - H/4)**2 + (Y - 3*W/4)**2) / (0.08 * H**2))  # Source 2
    )
    S = S / S.max()  # Normalize

    # Generate ground truth data
    print(f"\n4. Solving complex PDE (this may take a minute)...")
    print(f"   Grid: {H}×{W}, Timepoints: {T}")

    g_observed = solver.solve_complex_chemotaxis(
        g_init, S, alpha_true, beta_true, gamma_true, K_true, num_steps=T
    )

    # Statistics
    print(f"\n5. Data statistics:")
    print(f"   Initial mass: {g_init.sum():.4f}")
    print(f"   Final mass:   {g_observed[:, :, -1].sum():.4f}")
    print(f"   Mass change:  {(g_observed[:, :, -1].sum() - g_init.sum()):.4f}")
    print(f"   Max density:  {g_observed.max():.4f}")
    print(f"   Min density:  {g_observed.min():.4f}")

    # Create visualization
    print(f"\n6. Creating visualization...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: Cell density at different times
    times = [0, T//3, 2*T//3, T-1]
    for i, t in enumerate(times):
        ax = axes[0, i]
        im = ax.imshow(g_observed[:, :, t], cmap='viridis', aspect='auto')
        ax.set_title(f'Cell Density t={t}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2: Analysis
    ax = axes[1, 0]
    ax.imshow(S, cmap='hot', aspect='auto')
    ax.set_title('Chemoattractant Field')
    ax.axis('off')

    ax = axes[1, 1]
    total_mass = np.sum(g_observed, axis=(0, 1))
    ax.plot(total_mass, 'b-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Mass')
    ax.set_title('Mass Evolution')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    mean_density = np.mean(g_observed, axis=(0, 1))
    ax.plot(mean_density, 'g-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean Density')
    ax.set_title('Spatial Average')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 3]
    max_density = np.max(g_observed, axis=(0, 1))
    ax.plot(max_density, 'r-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Max Density')
    ax.set_title('Peak Density')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    viz_path = output_path / "complex_pde_overview.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved to: {viz_path}")

    # Create problem
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
            'dx': 1.0, 'dy': 1.0, 'dt': 0.01,
            'reference_pde': gt_equation,
            'description': 'Complex chemotaxis with logistic growth',
            'difficulty': 'hard'
        },
        gt_equation=gt_equation
    )

    # Save dataset
    dm = ChemotaxisDataModule(data_source="synthetic")
    dm.problems = {'complex_001': problem}

    dataset_path = output_path / "complex_chemotaxis_test.hdf5"
    dm.save_hdf5(str(dataset_path))
    print(f"\n7. Dataset saved to: {dataset_path}")

    # Also save as numpy for easy loading
    numpy_path = output_path / "complex_chemotaxis_test.npz"
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
    print(f"   Also saved numpy format: {numpy_path}")

    print(f"\n" + "="*70)
    print("COMPLEX TEST CASE CREATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nGround Truth PDE:")
    print(f"  {gt_equation}")
    print(f"\nTrue Parameters:")
    print(f"  α = {alpha_true}")
    print(f"  β = {beta_true}")
    print(f"  γ = {gamma_true}")
    print(f"  K = {K_true}")
    print(f"\nChallenge: Can the system discover all 4 parameters and the correct form?")

    return str(dataset_path)


if __name__ == "__main__":
    create_complex_test_case()
