#!/usr/bin/env python3
"""
Example: End-to-End PDE Discovery for Chemotaxis

This script demonstrates the complete workflow for discovering PDEs
from spatiotemporal imaging data using the dual-agent system.
"""

import numpy as np
from pathlib import Path
import argparse

# Import PDE discovery components
from bench.pde_solver import PDESolver, PDEConfig
from bench.pde_visualization import PDEVisualizer
from bench.pde_experience_buffer import PDEExperienceBuffer
from bench.pde_datamodule import ChemotaxisDataModule, create_chemotaxis_datamodule
from bench.pde_agents import SimplePDEDiscoverySystem, PDEDiscoverySystem


def example_basic_solver():
    """Example 1: Basic PDE solver usage"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic PDE Solver")
    print("="*70)

    # Create test data
    print("\n1. Generating synthetic chemotaxis data...")
    data = create_chemotaxis_datamodule()

    g_init = data['g_init']
    S = data['S']
    g_observed = data['g_observed']
    alpha_true = data['alpha_true']

    print(f"   ✓ Grid size: {g_init.shape}")
    print(f"   ✓ Timepoints: {g_observed.shape[2]}")
    print(f"   ✓ True diffusion coefficient: {alpha_true}")

    # Initialize solver
    print("\n2. Solving reference PDE...")
    solver = PDESolver(PDEConfig(dx=1.0, dy=1.0, dt=0.01))

    # Solve with different alpha
    alpha_test = 0.4
    g_predicted = solver.solve_reference_pde(g_init, S, alpha_test, num_steps=g_observed.shape[2])

    # Compute metrics
    print("\n3. Computing metrics...")
    mse = solver.compute_spatiotemporal_loss(g_predicted, g_observed, 'mse')
    r2 = solver.compute_spatiotemporal_loss(g_predicted, g_observed, 'r2')
    nmse = solver.compute_spatiotemporal_loss(g_predicted, g_observed, 'nmse')

    print(f"   ✓ MSE:  {mse:.6f}")
    print(f"   ✓ R²:   {r2:.4f}")
    print(f"   ✓ NMSE: {nmse:.4f}")

    # Fit parameters
    print("\n4. Fitting PDE parameters...")
    pde_template = "∂g/∂t = α·Δg - ∇·(g∇(ln S))"
    fitted_params, loss = solver.fit_pde_parameters(
        pde_template, g_init, S, g_observed,
        param_bounds={'α': (0.01, 2.0)}
    )

    print(f"   ✓ Fitted α: {fitted_params['α']:.4f}")
    print(f"   ✓ True α:   {alpha_true:.4f}")
    print(f"   ✓ Error:    {abs(fitted_params['α'] - alpha_true):.4f}")

    return g_observed, g_predicted, S


def example_visualization(g_observed, g_predicted):
    """Example 2: Creating visualizations"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Visualization")
    print("="*70)

    visualizer = PDEVisualizer(figsize=(14, 10), dpi=100)

    # Create comprehensive visualization
    print("\n1. Creating comprehensive visualization...")
    metrics = {
        'mse': np.mean((g_predicted - g_observed)**2),
        'r2': 1 - np.sum((g_observed - g_predicted)**2) / np.sum((g_observed - np.mean(g_observed))**2),
        'nmse': np.mean((g_predicted - g_observed)**2) / np.var(g_observed),
        'mass_error': abs(g_predicted[:,:,-1].sum() - g_observed[:,:,-1].sum()) / g_observed[:,:,-1].sum() * 100
    }

    img = visualizer.create_comprehensive_plot(
        g_observed, g_predicted,
        equation_str="∂g/∂t = α·Δg - ∇·(g∇(ln S))",
        score=8.5,
        save_path="example_comprehensive.png"
    )
    print(f"   ✓ Saved to: example_comprehensive.png")

    # Create critique visualization
    print("\n2. Creating critique visualization...")
    img = visualizer.create_critique_visualization(
        g_observed, g_predicted,
        equation_str="∂g/∂t = α·Δg - ∇·(g∇(ln S))",
        metrics=metrics,
        save_path="example_critique.png"
    )
    print(f"   ✓ Saved to: example_critique.png")

    print(f"\n   Metrics:")
    for key, val in metrics.items():
        print(f"   - {key}: {val:.6f}")


def example_experience_buffer():
    """Example 3: Using experience buffer"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Experience Buffer")
    print("="*70)

    buffer = PDEExperienceBuffer(max_size=50, diversity_threshold=0.3)

    # Simulate adding experiences during discovery
    print("\n1. Adding experiences...")
    experiences = [
        ("∂g/∂t = α·Δg", 4.2, {'mse': 0.08, 'r2': 0.62}, "Pure diffusion model"),
        ("∂g/∂t = -∇·(g∇S)", 5.8, {'mse': 0.04, 'r2': 0.78}, "Pure chemotaxis model"),
        ("∂g/∂t = α·Δg - ∇·(g∇S)", 7.1, {'mse': 0.02, 'r2': 0.88}, "Linear chemotaxis"),
        ("∂g/∂t = α·Δg - ∇·(g∇(ln S))", 9.2, {'mse': 0.005, 'r2': 0.97}, "Logarithmic gradient sensing"),
        ("∂g/∂t = α·Δg - β·∇·(g∇(ln S))", 8.8, {'mse': 0.008, 'r2': 0.95}, "Weighted chemotaxis"),
    ]

    for eq, score, metrics, reasoning in experiences:
        buffer.add(
            equation=eq,
            score=score,
            metrics=metrics,
            reasoning=reasoning,
            visual_analysis=f"Visual analysis of {eq[:20]}...",
            suggestions="Consider parameter adjustments",
            parameters={'α': 0.5}
        )

    print(f"   ✓ Added {len(buffer)} experiences")

    # Retrieve top experiences
    print("\n2. Retrieving top experiences...")
    top_3 = buffer.get_top_k(k=3)
    for i, exp in enumerate(top_3, 1):
        print(f"   {i}. Score {exp.score:.1f}: {exp.equation}")

    # Get best
    best = buffer.get_best()
    print(f"\n3. Best equation:")
    print(f"   {best.equation}")
    print(f"   Score: {best.score:.2f}")
    print(f"   R²: {best.metrics['r2']:.4f}")

    # Format for prompting
    print("\n4. Generating prompt context...")
    context = buffer.format_for_prompt(k=3, include_visual=False)
    print(f"   ✓ Context length: {len(context)} characters")
    print(f"\n   Preview:")
    print(f"   {context[:300]}...")

    # Save buffer
    print("\n5. Saving buffer...")
    buffer.save("example_buffer.json")
    print(f"   ✓ Saved to: example_buffer.json")

    # Statistics
    print("\n6. Buffer statistics:")
    stats = buffer.get_statistics()
    for key, val in stats.items():
        print(f"   - {key}: {val}")


def example_datamodule():
    """Example 4: Using datamodule"""
    print("\n" + "="*70)
    print("EXAMPLE 4: DataModule")
    print("="*70)

    # Load synthetic data
    print("\n1. Loading synthetic data...")
    dm = ChemotaxisDataModule(data_source="synthetic")
    problems = dm.load()

    print(f"   ✓ Loaded {len(problems)} problems")

    # Get first problem
    problem_id = list(problems.keys())[0]
    problem = problems[problem_id]

    print(f"\n2. Problem details:")
    print(f"   - ID: {problem_id}")
    print(f"   - g_init shape: {problem.g_init.shape}")
    print(f"   - S shape: {problem.S.shape}")
    print(f"   - g_observed shape: {problem.g_observed.shape}")
    print(f"   - Ground truth: {problem.gt_equation}")

    # Convert to SEDTask
    print(f"\n3. Converting to SEDTask format...")
    task = problem.to_sed_task()
    print(f"   ✓ Task name: {task.name}")
    print(f"   ✓ Symbols: {', '.join(task.symbols)}")
    print(f"   ✓ Description length: {len(task.desc)} chars")

    # Save as HDF5
    print(f"\n4. Saving as HDF5...")
    dm.save_hdf5("example_chemotaxis.hdf5")
    print(f"   ✓ Saved to: example_chemotaxis.hdf5")

    # Load back
    print(f"\n5. Loading from HDF5...")
    dm_loaded = ChemotaxisDataModule(data_source="hdf5", data_path="example_chemotaxis.hdf5")
    problems_loaded = dm_loaded.load()
    print(f"   ✓ Loaded {len(problems_loaded)} problems")

    return problem


def example_full_discovery(problem, use_autogen=False):
    """Example 5: Full discovery pipeline"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Full Discovery Pipeline")
    print("="*70)

    if use_autogen:
        print("\n⚠ AutoGen mode requires:")
        print("  1. pip install pyautogen")
        print("  2. vLLM server running: vllm serve Qwen/Qwen2-VL-7B-Instruct")
        print("\nSkipping AutoGen example. Use --autogen flag to enable.")
        return

    # Use simplified version
    print("\n1. Initializing discovery system (simplified mode)...")
    system = SimplePDEDiscoverySystem(
        max_iterations=3,  # Small for demo
        work_dir="./example_discovery"
    )
    print(f"   ✓ System initialized")

    print("\n2. Running discovery...")
    print("   Note: This simplified version tests predefined candidates.")
    print("   For full AI-driven discovery, use PDEDiscoverySystem with AutoGen.\n")

    results = system.discover(
        g_init=problem.g_init,
        S=problem.S,
        g_observed=problem.g_observed,
        verbose=True
    )

    print("\n3. Discovery results:")
    if results['success']:
        print(f"   ✓ Best equation: {results['best_equation']}")
        print(f"   ✓ Best score (R²×10): {results['best_score']:.2f}")

        print(f"\n4. All tested equations:")
        for i, res in enumerate(results.get('results', []), 1):
            print(f"   {i}. {res['equation']}")
            print(f"      MSE: {res['metrics']['mse']:.6f}, R²: {res['metrics']['r2']:.4f}")
    else:
        print("   ✗ Discovery failed")


def main():
    """Run all examples"""
    parser = argparse.ArgumentParser(description="PDE Discovery Examples")
    parser.add_argument('--autogen', action='store_true', help='Use AutoGen for full discovery')
    parser.add_argument('--example', type=int, choices=[1,2,3,4,5], help='Run specific example')
    args = parser.parse_args()

    print("="*70)
    print("PDE DISCOVERY SYSTEM - END-TO-END EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates all components of the PDE discovery system:")
    print("1. Basic PDE solver usage")
    print("2. Creating visualizations")
    print("3. Experience buffer management")
    print("4. Data loading and management")
    print("5. Full discovery pipeline")

    # Run examples
    if args.example is None or args.example == 1:
        g_observed, g_predicted, S = example_basic_solver()
    else:
        # Load data for other examples
        data = create_chemotaxis_datamodule()
        g_observed = data['g_observed']
        g_predicted = g_observed + np.random.normal(0, 0.01, g_observed.shape)

    if args.example is None or args.example == 2:
        example_visualization(g_observed, g_predicted)

    if args.example is None or args.example == 3:
        example_experience_buffer()

    if args.example is None or args.example == 4:
        problem = example_datamodule()
    else:
        dm = ChemotaxisDataModule(data_source="synthetic")
        problems = dm.load()
        problem = list(problems.values())[0]

    if args.example is None or args.example == 5:
        example_full_discovery(problem, use_autogen=args.autogen)

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED!")
    print("="*70)
    print("\nGenerated files:")
    print("  - example_comprehensive.png")
    print("  - example_critique.png")
    print("  - example_buffer.json")
    print("  - example_chemotaxis.hdf5")
    print("  - example_discovery/ (directory)")
    print("\nNext steps:")
    print("  1. Review generated visualizations")
    print("  2. Examine experience_buffer.json")
    print("  3. Try with real data: modify example_datamodule()")
    print("  4. Enable AutoGen: python example_pde_discovery.py --autogen")


if __name__ == "__main__":
    main()
