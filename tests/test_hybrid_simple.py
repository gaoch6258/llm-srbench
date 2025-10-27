#!/usr/bin/env python3
"""
Simple test script for hybrid solver (no pytest dependency)
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.pde_symbolic_solver import SymbolicPDEParser, SymbolicPDEEvaluator
from bench.pde_hybrid_solver import PDEHybridSolver, SolverTier
import numpy as np
from scipy.ndimage import laplace
g = np.zeros((3, 4, 5))
g.shape
laplace(g)

def test_parser():
    """Test symbolic parser"""
    print("\n" + "="*70)
    print("TEST 1: Symbolic Parser")
    print("="*70)

    parser = SymbolicPDEParser()

    # Test 1: Simple diffusion
    print("\n1. Parsing: ∂g/∂t = α·Δg")
    structure = parser.parse("∂g/∂t = α·Δg")
    print(f"   Valid: {structure.is_valid}")
    print(f"   Operators: {len(structure.spatial_operators)}")
    print(f"   Parameters: {structure.parameters}")
    print(f"   Reaction terms: {len(structure.reaction_terms)}")

    if structure.is_valid and len(structure.spatial_operators) >= 1:
        print("   ✅ PASS")
    else:
        print("   ❌ FAIL")
        print(f"   Details: {structure}")

    # Test 2: Chemotaxis
    print("\n2. Parsing: ∂g/∂t = α·Δg - β·∇·(g∇(ln S))")
    structure = parser.parse("∂g/∂t = α·Δg - β·∇·(g∇(ln S))")
    print(f"   Valid: {structure.is_valid}")
    print(f"   Operators: {len(structure.spatial_operators)}")
    print(f"   Parameters: {structure.parameters}")

    if structure.is_valid:
        print("   ✅ PASS")
    else:
        print("   ❌ FAIL")


def test_symbolic_solver():
    """Test symbolic solver"""
    print("\n" + "="*70)
    print("TEST 2: Symbolic Solver")
    print("="*70)

    # Create problem
    H, W = 32, 32
    T = 20
    x = np.linspace(0, H-1, H)
    y = np.linspace(0, W-1, W)
    X, Y = np.meshgrid(x, y)
    g_init = np.exp(-((X - H/2)**2 + (Y - W/2)**2) / (0.1 * H**2))
    S = np.ones((H, W))

    solver = SymbolicPDEEvaluator(dx=1.0, dy=1.0, dt=0.01)

    # Test 1: Diffusion
    print("\n1. Solving: ∂g/∂t = α·Δg")
    try:
        solution, info = solver.solve(
            pde_str="∂g/∂t = α·Δg",
            g_init=g_init,
            S=S,
            param_values={'α': 0.5},
            num_steps=T
        )

        if solution is not None and solution.shape == (H, W, T):
            print(f"   Solution shape: {solution.shape}")
            print(f"   Range: [{solution.min():.4f}, {solution.max():.4f}]")
            print(f"   Method: {info['method']}")
            print("   ✅ PASS")
        else:
            print("   ❌ FAIL - Invalid solution")
    except Exception as e:
        print(f"   ❌ FAIL - {e}")

    # Test 2: Chemotaxis
    print("\n2. Solving: ∂g/∂t = α·Δg - β·∇·(g∇(ln S))")
    S_grad = np.exp(0.01 * X + 0.01 * Y)
    try:
        solution, info = solver.solve(
            pde_str="∂g/∂t = α·Δg - β·∇·(g∇(ln S))",
            g_init=g_init,
            S=S_grad,
            param_values={'α': 0.5, 'β': 1.0},
            num_steps=T
        )

        if solution is not None and solution.shape == (H, W, T):
            print(f"   Solution shape: {solution.shape}")
            print(f"   Range: [{solution.min():.4f}, {solution.max():.4f}]")
            print("   ✅ PASS")
        else:
            print("   ❌ FAIL - Invalid solution")
    except Exception as e:
        print(f"   ❌ FAIL - {e}")


def test_hybrid_solver():
    """Test hybrid solver"""
    print("\n" + "="*70)
    print("TEST 3: Hybrid Solver")
    print("="*70)

    # Create problem
    H, W = 32, 32
    T = 20
    x = np.linspace(0, H-1, H)
    y = np.linspace(0, W-1, W)
    X, Y = np.meshgrid(x, y)
    g_init = np.exp(-((X - H/2)**2 + (Y - W/2)**2) / (0.1 * H**2))
    S = np.exp(0.01 * X + 0.01 * Y)

    solver = PDEHybridSolver(
        dx=1.0, dy=1.0, dt=0.01,
        prefer_symbolic=True,
        enable_codegen=False,  # Disable for quick test
        verbose=False
    )

    # Test routing
    print("\n1. Testing routing to symbolic solver")
    result = solver.solve(
        pde_str="∂g/∂t = α·Δg",
        g_init=g_init,
        S=S,
        param_values={'α': 0.5},
        num_steps=T
    )

    if result.success:
        print(f"   Success: {result.success}")
        print(f"   Tier: {result.tier_used.value}")
        print(f"   Time: {result.execution_time:.3f}s")

        if result.tier_used == SolverTier.SYMBOLIC:
            print("   ✅ PASS - Routed to symbolic")
        else:
            print(f"   ⚠️  WARN - Routed to {result.tier_used.value} instead")
    else:
        print(f"   ❌ FAIL - {result.error_message}")

    # Test multiple PDEs
    print("\n2. Testing multiple PDE forms")
    test_cases = [
        ("∂g/∂t = α·Δg", {'α': 0.5}),
        ("∂g/∂t = α·Δg + β·g", {'α': 0.5, 'β': 0.1}),
        ("∂g/∂t = α·Δg - β·g·S", {'α': 0.5, 'β': 0.1}),
    ]

    successes = 0
    for pde, params in test_cases:
        result = solver.solve(pde, g_init, S, params, T)
        status = "✓" if result.success else "✗"
        tier = result.tier_used.value if result.success else "FAIL"
        print(f"   {status} {pde[:40]:40s} [{tier}]")
        if result.success:
            successes += 1

    if successes >= 2:
        print(f"   ✅ PASS - {successes}/{len(test_cases)} succeeded")
    else:
        print(f"   ❌ FAIL - Only {successes}/{len(test_cases)} succeeded")


def main():
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "HYBRID PDE SOLVER - TESTS" + " "*28 + "║")
    print("╚" + "="*68 + "╝")

    try:
        test_parser()
        test_symbolic_solver()
        test_hybrid_solver()

        print("\n" + "="*70)
        print("ALL TESTS COMPLETE")
        print("="*70)
        print("\n✅ If you see mostly PASS above, the hybrid solver is working!")
        print("✅ Some failures are expected for edge cases\n")

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
