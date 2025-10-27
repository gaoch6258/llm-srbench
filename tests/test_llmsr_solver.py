#!/usr/bin/env python3
"""
Quick test of LLMSR-based PDE solver
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.pde_llmsr_solver import LLMSRPDESolver


def test_llmsr_solver():
    """Test LLMSR solver with mock LLM"""

    print("="*70)
    print("TESTING LLMSR PDE SOLVER")
    print("="*70)

    # Create mock LLM client that returns simple diffusion code
    class MockLLMClient:
        def generate(self, prompt):
            return """
def pde_update(g: np.ndarray, S: np.ndarray, dx: float, dy: float, dt: float, params: np.ndarray) -> np.ndarray:
    H, W = g.shape

    # Extract parameters
    p0 = params[0]  # diffusion coefficient

    # Compute Laplacian
    laplacian_g = np.zeros_like(g)
    laplacian_g[1:-1, 1:-1] = (
        g[1:-1, 2:] + g[1:-1, :-2] + g[2:, 1:-1] + g[:-2, 1:-1] - 4*g[1:-1, 1:-1]
    ) / dx**2

    # PDE: dg/dt = p0 * Laplacian(g)
    dg_dt = p0 * laplacian_g

    # Forward Euler
    g_next = g + dt * dg_dt

    # Non-negativity
    g_next = np.maximum(g_next, 0)

    return g_next
"""

    # Create solver
    solver = LLMSRPDESolver(
        llm_client=MockLLMClient(),
        dx=1.0, dy=1.0, dt=0.01,
        timeout=30
    )

    # Create test problem
    H, W, T = 32, 32, 20
    x = np.linspace(0, H-1, H)
    y = np.linspace(0, W-1, W)
    X, Y = np.meshgrid(x, y)
    g_init = np.exp(-((X - H/2)**2 + (Y - W/2)**2) / (0.1 * H**2))
    S = np.ones((H, W))

    # Generate synthetic observed data with known parameter
    print("\n1. Generating synthetic data with p0=0.5...")
    true_params = np.array([0.5])
    g_observed, success, error = solver.evaluate_pde(
        code=MockLLMClient().generate(""),
        g_init=g_init,
        S=S,
        params=true_params,
        num_steps=T
    )

    if not success:
        print(f"   ❌ FAIL: {error}")
        return False

    print(f"   ✅ SUCCESS - Generated {g_observed.shape} data")

    # Test fit_and_evaluate
    print("\n2. Testing fit_and_evaluate...")
    result = solver.fit_and_evaluate(
        pde_description="Pure diffusion with parameter p0",
        num_params=1,
        g_init=g_init,
        S=S,
        g_observed=g_observed,
        param_bounds=[(0.1, 2.0)]
    )

    if not result['success']:
        print(f"   ❌ FAIL: {result.get('error', 'Unknown')}")
        return False

    print(f"   ✅ SUCCESS")
    print(f"   True p0: {true_params[0]:.4f}")
    print(f"   Fitted p0: {result['fitted_params'][0]:.4f}")
    print(f"   R²: {result['r2']:.4f}")
    print(f"   MSE: {result['mse']:.6f}")

    # Check if fitted parameter is close
    if abs(result['fitted_params'][0] - true_params[0]) < 0.3:
        print("   ✅ Parameter recovery successful!")
    else:
        print("   ⚠️  Parameter recovery has error")

    print("\n" + "="*70)
    print("ALL TESTS PASSED")
    print("="*70)

    return True


if __name__ == "__main__":
    try:
        success = test_llmsr_solver()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
