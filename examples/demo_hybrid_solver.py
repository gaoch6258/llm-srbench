#!/usr/bin/env python3
"""
Quick Example: Using the Hybrid PDE Solver

This script demonstrates the hybrid solver with various PDE forms.
"""

import numpy as np
import sys
from pathlib import Path

# Add bench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.pde_hybrid_solver import PDEHybridSolver, SolverTier


def create_test_problem(H=32, W=32):
    """Create a simple test problem"""
    x = np.linspace(0, H-1, H)
    y = np.linspace(0, W-1, W)
    X, Y = np.meshgrid(x, y)

    # Initial condition: Gaussian blob
    g_init = np.exp(-((X - H/2)**2 + (Y - W/2)**2) / (0.1 * H**2))

    # Signal field: gradient
    S = np.exp(0.01 * X + 0.01 * Y)

    return g_init, S


def test_pde(solver, pde_str, params, g_init, S, T=20):
    """Test a single PDE"""
    print(f"\nTesting: {pde_str}")
    print(f"Parameters: {params}")

    result = solver.solve(pde_str, g_init, S, params, T)

    if result.success:
        print(f"✅ SUCCESS")
        print(f"   Tier: {result.tier_used.value}")
        print(f"   Time: {result.execution_time:.3f}s")
        print(f"   Shape: {result.solution.shape}")
        print(f"   Range: [{result.solution.min():.4f}, {result.solution.max():.4f}]")

        # Check mass conservation
        mass_init = g_init.sum()
        mass_final = result.solution[:, :, -1].sum()
        mass_change = (mass_final - mass_init) / mass_init * 100
        print(f"   Mass change: {mass_change:.2f}%")
    else:
        print(f"❌ FAILED")
        print(f"   Error: {result.error_message}")
        print(f"   Tier attempted: {result.tier_used.value}")

    return result


def main():
    print("="*70)
    print("HYBRID PDE SOLVER - DEMO")
    print("="*70)

    # Create solver
    print("\nInitializing hybrid solver...")
    solver = PDEHybridSolver(
        dx=1.0, dy=1.0, dt=0.01,
        boundary_condition="periodic",
        prefer_symbolic=True,
        enable_codegen=True,
        verbose=True  # Show routing decisions
    )

    # Create test problem
    g_init, S = create_test_problem(H=32, W=32)
    T = 20

    print(f"\nProblem setup:")
    print(f"  Grid: {g_init.shape[0]}×{g_init.shape[1]}")
    print(f"  Time steps: {T}")
    print(f"  dt: {solver.dt}")

    # Test cases
    test_cases = [
        # Tier 1: Symbolic solver (should be fast)
        ("∂g/∂t = α·Δg", {'α': 0.5}),
        ("∂g/∂t = α·Δg - β·∇·(g∇(ln S))", {'α': 0.5, 'β': 1.0}),
        ("∂g/∂t = α·Δg + γ·g(1-g/K)", {'α': 0.5, 'γ': 0.1, 'K': 5.0}),
        ("∂g/∂t = α·Δg + β·g²", {'α': 0.5, 'β': 0.1}),
        ("∂g/∂t = α·Δg - β·g·S", {'α': 0.5, 'β': 0.1}),

        # Tier 2: Code generation (may be slower, more flexible)
        # Note: Template may not perfectly match these exotic forms
        ("∂g/∂t = α·Δg + β·g² - γ·g·S", {'α': 0.5, 'β': 0.1, 'γ': 0.05}),
    ]

    print("\n" + "="*70)
    print("RUNNING TESTS")
    print("="*70)

    results = []
    for pde_str, params in test_cases:
        result = test_pde(solver, pde_str, params, g_init, S, T)
        results.append((pde_str, result))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    symbolic_count = sum(1 for _, r in results if r.success and r.tier_used == SolverTier.SYMBOLIC)
    codegen_count = sum(1 for _, r in results if r.success and r.tier_used == SolverTier.CODEGEN)
    legacy_count = sum(1 for _, r in results if r.success and r.tier_used == SolverTier.LEGACY)
    failed_count = sum(1 for _, r in results if not r.success)

    print(f"\nTotal tests: {len(results)}")
    print(f"  Tier 1 (Symbolic): {symbolic_count}")
    print(f"  Tier 2 (CodeGen): {codegen_count}")
    print(f"  Tier 3 (Legacy): {legacy_count}")
    print(f"  Failed: {failed_count}")

    # Statistics
    stats = solver.get_statistics()
    print(f"\nSolver statistics:")
    print(f"  Symbolic success rate: {stats['symbolic_rate']:.1%}")
    print(f"  CodeGen success rate: {stats['codegen_rate']:.1%}")
    print(f"  Legacy success rate: {stats['legacy_rate']:.1%}")

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)

    # Test parameter fitting
    print("\n" + "="*70)
    print("BONUS: PARAMETER FITTING DEMO")
    print("="*70)

    print("\nGenerating synthetic data with known parameters...")
    true_params = {'α': 0.5}
    result_true = solver.solve("∂g/∂t = α·Δg", g_init, S, true_params, T)

    if result_true.success:
        g_observed = result_true.solution

        print(f"Fitting parameters to recover true values...")
        param_bounds = {'α': (0.1, 2.0)}

        fitted_params, loss = solver.fit_parameters(
            pde_str="∂g/∂t = α·Δg",
            g_init=g_init,
            S=S,
            g_observed=g_observed,
            param_bounds=param_bounds
        )

        print(f"\nResults:")
        print(f"  True α: {true_params['α']:.4f}")
        print(f"  Fitted α: {fitted_params['α']:.4f}")
        print(f"  Error: {abs(fitted_params['α'] - true_params['α']):.4f}")
        print(f"  Loss (MSE): {loss:.6f}")

        if abs(fitted_params['α'] - true_params['α']) < 0.1:
            print("  ✅ Parameter fitting successful!")
        else:
            print("  ⚠️  Parameter fitting has large error")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
