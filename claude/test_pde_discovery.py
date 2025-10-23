#!/usr/bin/env python3
"""
Test script for PDE Discovery System

Tests all components:
1. PDE Solver
2. Visualization
3. Experience Buffer
4. Datamodule
5. Full discovery system (simplified version)
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import our modules
from bench.pde_solver import PDESolver, PDEConfig, create_chemotaxis_datamodule
from bench.pde_visualization import PDEVisualizer
from bench.pde_experience_buffer import PDEExperienceBuffer
from bench.pde_datamodule import ChemotaxisDataModule, create_test_dataset
from bench.pde_agents import SimplePDEDiscoverySystem


def test_pde_solver():
    """Test PDE solver component"""
    print("\n" + "="*60)
    print("TEST 1: PDE Solver")
    print("="*60)

    # Create test data
    solver = PDESolver(PDEConfig(dx=1.0, dy=1.0, dt=0.01, max_iterations=50))

    # Small test case
    H, W = 32, 32
    T = 30

    x = np.linspace(0, H-1, H)
    y = np.linspace(0, W-1, W)
    X, Y = np.meshgrid(x, y)

    g_init = np.exp(-((X - H/2)**2 + (Y - W/2)**2) / (0.1 * H**2))
    S = np.exp(0.01 * X + 0.01 * Y)

    print(f"\nSolving reference PDE with α=0.5")
    print(f"Grid: {H}x{W}, Timepoints: {T}")

    # Solve reference PDE
    alpha = 0.5
    g_history = solver.solve_reference_pde(g_init, S, alpha, num_steps=T)

    print(f"✓ Solution shape: {g_history.shape}")
    print(f"✓ Mass initial: {g_init.sum():.4f}")
    print(f"✓ Mass final: {g_history[:, :, -1].sum():.4f}")
    print(f"✓ Mass change: {(g_history[:, :, -1].sum() - g_init.sum()) / g_init.sum() * 100:.2f}%")

    # Test pure diffusion
    print(f"\nSolving pure diffusion PDE")
    g_diffusion = solver.solve_diffusion(g_init, alpha, num_steps=T)
    print(f"✓ Diffusion solution shape: {g_diffusion.shape}")

    # Test loss computation
    mse = solver.compute_spatiotemporal_loss(g_history, g_history, 'mse')
    r2 = solver.compute_spatiotemporal_loss(g_history, g_history, 'r2')
    print(f"\n✓ Self-comparison MSE: {mse:.2e} (should be ~0)")
    print(f"✓ Self-comparison R²: {r2:.4f} (should be ~1)")

    # Test parameter fitting
    print(f"\nTesting parameter fitting...")
    pde_template = "∂g/∂t = α·Δg - ∇·(g∇(ln S))"
    fitted_params, loss = solver.fit_pde_parameters(
        pde_template, g_init, S, g_history,
        param_bounds={'α': (0.01, 2.0)}
    )
    print(f"✓ Fitted α: {fitted_params.get('α', 0):.4f} (true: {alpha:.4f})")
    print(f"✓ Fitting loss: {loss:.2e}")

    return g_history, S


def test_visualization(g_observed, S):
    """Test visualization component"""
    print("\n" + "="*60)
    print("TEST 2: Visualization")
    print("="*60)

    visualizer = PDEVisualizer(figsize=(14, 10), dpi=80)

    # Create predicted data (add some noise)
    g_predicted = g_observed + np.random.normal(0, 0.01, g_observed.shape)

    print("\nCreating comprehensive visualization...")
    img = visualizer.create_comprehensive_plot(
        g_observed, g_predicted,
        equation_str="∂g/∂t = α·Δg - ∇·(g∇(ln S))",
        score=8.5,
        save_path="test_comprehensive.png"
    )
    print(f"✓ Comprehensive plot created: test_comprehensive.png")
    print(f"✓ Image size: {img.size}")

    print("\nCreating critique visualization...")
    metrics = {
        'mse': 0.001,
        'r2': 0.95,
        'nmse': 0.05,
        'mass_error': 2.3
    }
    img = visualizer.create_critique_visualization(
        g_observed, g_predicted,
        equation_str="∂g/∂t = α·Δg - ∇·(g∇(ln S))",
        metrics=metrics,
        save_path="test_critique.png"
    )
    print(f"✓ Critique plot created: test_critique.png")

    print("\nCreating simple comparison...")
    img = visualizer.create_simple_comparison(g_observed, g_predicted, timepoint=-1)
    print(f"✓ Simple comparison created")


def test_experience_buffer():
    """Test experience buffer component"""
    print("\n" + "="*60)
    print("TEST 3: Experience Buffer")
    print("="*60)

    buffer = PDEExperienceBuffer(max_size=10, diversity_threshold=0.3)

    print(f"\nAdding experiences...")
    # Add some test experiences
    experiences = [
        ("∂g/∂t = α·Δg", 3.5, {'mse': 0.1, 'r2': 0.5}),
        ("∂g/∂t = α·Δg - ∇·(g∇S)", 6.2, {'mse': 0.05, 'r2': 0.75}),
        ("∂g/∂t = α·Δg - ∇·(g∇(ln S))", 8.7, {'mse': 0.01, 'r2': 0.95}),
        ("∂g/∂t = α·Δg - β·∇·(g∇S)", 7.1, {'mse': 0.03, 'r2': 0.85}),
    ]

    for eq, score, metrics in experiences:
        buffer.add(
            equation=eq,
            score=score,
            metrics=metrics,
            visual_analysis=f"Analysis of {eq[:20]}...",
            reasoning=f"Testing equation {eq[:20]}...",
            suggestions="Try different parameters",
            parameters={'α': 0.5}
        )

    print(f"✓ Added {len(buffer)} experiences")
    print(f"✓ Buffer state: {buffer}")

    # Test retrieval
    top3 = buffer.get_top_k(k=3)
    print(f"\n✓ Top 3 equations:")
    for i, exp in enumerate(top3, 1):
        print(f"  {i}. {exp.equation[:40]}... (score: {exp.score:.2f})")

    best = buffer.get_best()
    print(f"\n✓ Best equation: {best.equation}")
    print(f"✓ Best score: {best.score:.2f}")

    # Test prompt formatting
    prompt_context = buffer.format_for_prompt(k=2)
    print(f"\n✓ Prompt context length: {len(prompt_context)} chars")
    print(f"✓ First 200 chars: {prompt_context[:200]}...")

    # Test save/load
    buffer.save("test_buffer.json")
    print(f"\n✓ Buffer saved to test_buffer.json")

    loaded_buffer = PDEExperienceBuffer.load("test_buffer.json")
    print(f"✓ Buffer loaded: {loaded_buffer}")

    # Test statistics
    stats = buffer.get_statistics()
    print(f"\n✓ Buffer statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val}")


def test_datamodule():
    """Test datamodule component"""
    print("\n" + "="*60)
    print("TEST 4: DataModule")
    print("="*60)

    # Test synthetic data
    print("\nLoading synthetic data...")
    dm_synth = ChemotaxisDataModule(data_source="synthetic")
    problems = dm_synth.load()
    print(f"✓ Loaded {len(problems)} synthetic problems")

    # Get first problem
    problem_id = list(problems.keys())[0]
    problem = problems[problem_id]
    print(f"\n✓ Problem: {problem_id}")
    print(f"✓ g_init shape: {problem.g_init.shape}")
    print(f"✓ S shape: {problem.S.shape}")
    print(f"✓ g_observed shape: {problem.g_observed.shape}")
    print(f"✓ GT equation: {problem.gt_equation}")

    # Test conversion to SEDTask
    task = problem.to_sed_task()
    print(f"\n✓ Converted to SEDTask: {task.name}")
    print(f"✓ Symbols: {task.symbols}")
    print(f"✓ Description length: {len(task.desc)} chars")

    # Test save/load HDF5
    print("\nTesting HDF5 save/load...")
    dm_synth.save_hdf5("test_chemotaxis.hdf5")
    print(f"✓ Saved to test_chemotaxis.hdf5")

    dm_loaded = ChemotaxisDataModule(data_source="hdf5", data_path="test_chemotaxis.hdf5")
    problems_loaded = dm_loaded.load()
    print(f"✓ Loaded {len(problems_loaded)} problems from HDF5")

    # Verify data matches
    problem_loaded = problems_loaded[problem_id]
    assert np.allclose(problem.g_init, problem_loaded.g_init), "Data mismatch!"
    print(f"✓ Data verification passed")

    # Create test dataset
    print("\nCreating test dataset...")
    test_path = create_test_dataset(output_dir="./test_data")
    print(f"✓ Test dataset created at {test_path}")

    return problem


def test_discovery_system(problem):
    """Test full discovery system (simplified)"""
    print("\n" + "="*60)
    print("TEST 5: Discovery System (Simplified)")
    print("="*60)

    print("\nInitializing discovery system...")
    system = SimplePDEDiscoverySystem(
        max_iterations=2,
        work_dir="./test_discovery"
    )
    print(f"✓ System initialized")
    print(f"✓ Work directory: {system.work_dir}")

    print("\nRunning discovery...")
    results = system.discover(
        g_init=problem.g_init,
        S=problem.S,
        g_observed=problem.g_observed,
        verbose=True
    )

    print(f"\n✓ Discovery completed")
    print(f"✓ Success: {results['success']}")
    print(f"✓ Best equation: {results.get('best_equation', 'None')}")
    print(f"✓ Best score: {results.get('best_score', 0):.4f}")

    if results['success']:
        print(f"\n✓ Results summary:")
        for res in results.get('results', []):
            print(f"  - {res['equation']}: R²={res['metrics']['r2']:.4f}")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PDE DISCOVERY SYSTEM - COMPREHENSIVE TEST")
    print("="*60)

    try:
        # Test 1: PDE Solver
        g_history, S = test_pde_solver()

        # Test 2: Visualization
        test_visualization(g_history, S)

        # Test 3: Experience Buffer
        test_experience_buffer()

        # Test 4: DataModule
        problem = test_datamodule()

        # Test 5: Discovery System
        test_discovery_system(problem)

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)

        print("\nGenerated files:")
        print("  - test_comprehensive.png")
        print("  - test_critique.png")
        print("  - test_buffer.json")
        print("  - test_chemotaxis.hdf5")
        print("  - test_data/chemotaxis_test.hdf5")
        print("  - test_discovery/ (directory)")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
