#!/usr/bin/env python3
"""
PDE Discovery - Full Pipeline Integration

Runs PDE discovery with LLM-SR style architecture using vLLM backend.
Integrates with existing eval.py pattern for consistency.
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np

from bench.pde_datamodule import ChemotaxisDataModule
from bench.pde_solver import PDESolver, PDEConfig
from bench.pde_visualization import PDEVisualizer
from bench.pde_experience_buffer import PDEExperienceBuffer
from bench.pde_prompts import (
    create_generator_prompt,
    extract_equation_from_response,
    extract_reasoning_from_response
)


class LLMSRPDEDiscovery:
    """
    PDE Discovery system following LLM-SR architecture

    Similar to LLMSRSearcher but specialized for spatiotemporal PDEs
    """

    def __init__(
        self,
        api_base: str = "http://localhost:10005/v1",
        api_model: str = "/mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct",
        max_iterations: int = 8000,
        samples_per_prompt: int = 4,
        convergence_threshold: float = 0.98,
        plateau_patience: int = 50,
        output_dir: str = "./logs/pde_discovery",
        solver_config: PDEConfig = None
    ):
        self.api_base = api_base
        self.api_model = api_model
        self.max_iterations = max_iterations
        self.samples_per_prompt = samples_per_prompt
        self.convergence_threshold = convergence_threshold
        self.plateau_patience = plateau_patience
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.solver = PDESolver(solver_config or PDEConfig())
        self.visualizer = PDEVisualizer()
        self.buffer = PDEExperienceBuffer(max_size=200)

        # State
        self.best_score = -float('inf')
        self.best_equation = None
        self.plateau_counter = 0
        self.iteration = 0

        # Setup LLM client
        self._setup_llm_client()

    def _setup_llm_client(self):
        """Setup OpenAI-compatible client for vLLM"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key="EMPTY",
                base_url=self.api_base
            )
            print(f"✓ Connected to LLM at {self.api_base}")
        except ImportError:
            print("Warning: openai package not installed. Install with: pip install openai")
            self.client = None
        except Exception as e:
            print(f"Warning: Could not connect to LLM: {e}")
            self.client = None

    def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Call LLM with prompt"""
        if self.client is None:
            # Fallback to reference equation for testing
            return """<equation>
∂g/∂t = α·Δg - β·∇·(g∇(ln S))
</equation>
<reasoning>
Testing fallback - LLM not available
</reasoning>"""

        try:
            response = self.client.chat.completions.create(
                model=self.api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Warning: LLM call failed: {e}")
            return ""

    def _generate_candidates(self, problem, data_summary, num_candidates: int = 4) -> list:
        """Generate PDE candidates using LLM"""
        # Get experience context
        experience_context = self.buffer.format_for_prompt(k=5, include_visual=False)

        # Create prompt
        problem_desc = f"Discover PDE for chemotaxis from {data_summary['shape']} imaging data"
        prompt = create_generator_prompt(
            problem_desc, data_summary, experience_context, self.iteration
        )

        candidates = []

        # Sample multiple candidates
        for i in range(num_candidates):
            response = self._call_llm(prompt, temperature=0.7 + i * 0.1)

            equation = extract_equation_from_response(response)
            reasoning = extract_reasoning_from_response(response)

            if equation:
                candidates.append({
                    'equation': equation,
                    'reasoning': reasoning
                })

        return candidates

    def _evaluate_candidate(self, equation: str, problem) -> dict:
        """Evaluate PDE candidate"""
        try:
            # Fit parameters
            param_bounds = {
                'α': (0.01, 2.0),
                'β': (0.01, 2.0),
                'γ': (0.001, 0.5),
                'K': (0.5, 5.0)
            }

            fitted_params, loss = self.solver.fit_pde_parameters(
                equation,
                problem.g_init,
                problem.S,
                problem.g_observed,
                param_bounds=param_bounds
            )

            # Solve with fitted parameters
            predicted, info = self.solver.evaluate_pde(
                equation,
                problem.g_init,
                problem.S,
                fitted_params,
                num_steps=problem.g_observed.shape[2]
            )

            # Compute metrics
            mse = float(self.solver.compute_spatiotemporal_loss(predicted, problem.g_observed, 'mse'))
            r2 = float(self.solver.compute_spatiotemporal_loss(predicted, problem.g_observed, 'r2'))
            nmse = float(self.solver.compute_spatiotemporal_loss(predicted, problem.g_observed, 'nmse'))

            # Mass conservation error
            obs_mass = np.sum(problem.g_observed, axis=(0, 1))
            pred_mass = np.sum(predicted, axis=(0, 1))
            mass_error = float(np.abs(pred_mass[-1] - obs_mass[-1]) / obs_mass[-1] * 100)

            # Composite score (R² weighted, penalize mass error)
            score = r2 * 10 * (1 - min(mass_error / 100, 0.5))

            return {
                'success': True,
                'predicted': predicted,
                'fitted_params': fitted_params,
                'metrics': {
                    'mse': mse,
                    'r2': r2,
                    'nmse': nmse,
                    'mass_error': mass_error
                },
                'score': score
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'score': -1.0
            }

    def discover(self, problem, verbose: bool = True):
        """Main discovery loop"""
        # Data summary
        H, W, T = problem.g_observed.shape
        data_summary = {
            'shape': f"({H}, {W}, {T})",
            'H': H, 'W': W, 'T': T,
            'dx': problem.metadata.get('dx', 1.0),
            'dy': problem.metadata.get('dy', 1.0),
            'dt': problem.metadata.get('dt', 0.01),
            'g_min': float(problem.g_observed.min()),
            'g_max': float(problem.g_observed.max()),
            'g_mean': float(problem.g_observed.mean()),
            'g_std': float(problem.g_observed.std()),
            'S_min': float(problem.S.min()),
            'S_max': float(problem.S.max()),
            'mass_initial': float(problem.g_init.sum()),
            'mass_final': float(problem.g_observed[:, :, -1].sum()),
            'mass_change_pct': float((problem.g_observed[:, :, -1].sum() - problem.g_init.sum()) / problem.g_init.sum() * 100),
            'spatial_spread': 'N/A'
        }

        start_time = time.time()

        if verbose:
            print("\n" + "="*70)
            print("PDE DISCOVERY - STARTING")
            print("="*70)
            print(f"Dataset: {data_summary['shape']}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Convergence threshold: {self.convergence_threshold}")
            print(f"Ground truth: {problem.gt_equation}")
            print("="*70)

        # Discovery loop
        for iteration in range(1, self.max_iterations + 1):
            self.iteration = iteration

            # Generate candidates
            candidates = self._generate_candidates(problem, data_summary, self.samples_per_prompt)

            # Evaluate each candidate
            for cand in candidates:
                equation = cand['equation']
                reasoning = cand['reasoning']

                if not equation:
                    continue

                # Evaluate
                result = self._evaluate_candidate(equation, problem)

                if result['success']:
                    score = result['score']
                    metrics = result['metrics']

                    # Add to buffer
                    self.buffer.add(
                        equation=equation,
                        score=score,
                        metrics=metrics,
                        visual_analysis="",
                        reasoning=reasoning,
                        suggestions="",
                        parameters=result['fitted_params']
                    )

                    # Check if best
                    if score > self.best_score:
                        self.best_score = score
                        self.best_equation = equation
                        self.plateau_counter = 0

                        if verbose and iteration % 10 == 0:
                            print(f"\nIter {iteration}: NEW BEST! Score={score:.4f}, R²={metrics['r2']:.4f}")
                            print(f"  Equation: {equation[:80]}...")
                            print(f"  Parameters: {result['fitted_params']}")

                        # Save visualization
                        if iteration % 100 == 0:
                            viz_path = self.output_dir / f"best_iter_{iteration:06d}.png"
                            self.visualizer.create_critique_visualization(
                                problem.g_observed,
                                result['predicted'],
                                equation,
                                metrics,
                                save_path=str(viz_path)
                            )

            # Update plateau counter
            if self.best_score == -float('inf') or score <= self.best_score:
                self.plateau_counter += 1

            # Progress report
            if verbose and iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")
                print(f"Best score: {self.best_score:.4f}")
                print(f"Plateau: {self.plateau_counter}/{self.plateau_patience}")
                print(f"Buffer size: {len(self.buffer)}")
                print(f"Elapsed: {elapsed:.1f}s")

            # Check convergence
            if self.best_score >= self.convergence_threshold * 10:
                if verbose:
                    print(f"\n✓ CONVERGED at iteration {iteration}!")
                break

            if self.plateau_counter >= self.plateau_patience:
                if verbose:
                    print(f"\n⚠ Plateau detected at iteration {iteration}")
                break

        # Final results
        total_time = time.time() - start_time

        results = {
            'success': self.best_equation is not None,
            'best_equation': self.best_equation,
            'best_score': float(self.best_score),
            'total_iterations': iteration,
            'total_time': total_time,
            'buffer_stats': self.buffer.get_statistics(),
            'gt_equation': problem.gt_equation,
            'gt_parameters': {
                k: v for k, v in problem.metadata.items()
                if k.endswith('_true')
            }
        }

        # Save results
        results_path = self.output_dir / "discovery_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Save buffer
        buffer_path = self.output_dir / "experience_buffer.json"
        self.buffer.save(str(buffer_path))

        if verbose:
            print("\n" + "="*70)
            print("DISCOVERY COMPLETE")
            print("="*70)
            print(f"Best equation: {self.best_equation}")
            print(f"Best score: {self.best_score:.4f}")
            print(f"Total iterations: {iteration}")
            print(f"Total time: {total_time:.1f}s")
            print(f"Results saved to: {results_path}")
            print("="*70)

        return results


def main():
    parser = argparse.ArgumentParser(description="PDE Discovery with LLM-SR")
    parser.add_argument('--dataset', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--api_base', type=str, default='http://localhost:10005/v1', help='vLLM API base URL')
    parser.add_argument('--api_model', type=str, default='/mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct', help='Model name/path')
    parser.add_argument('--max_iterations', type=int, default=8000, help='Maximum iterations')
    parser.add_argument('--samples_per_prompt', type=int, default=4, help='Samples per prompt')
    parser.add_argument('--output_dir', type=str, default='./logs/pde_discovery_run', help='Output directory')
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dm = ChemotaxisDataModule(data_source="hdf5", data_path=args.dataset)
    problems = dm.load()

    problem_id = list(problems.keys())[0]
    problem = problems[problem_id]

    print(f"✓ Loaded problem: {problem_id}")
    print(f"  Shape: {problem.g_observed.shape}")
    print(f"  GT equation: {problem.gt_equation}")

    # Initialize discovery system
    system = LLMSRPDEDiscovery(
        api_base=args.api_base,
        api_model=args.api_model,
        max_iterations=args.max_iterations,
        samples_per_prompt=args.samples_per_prompt,
        output_dir=args.output_dir
    )

    # Run discovery
    results = system.discover(problem, verbose=True)

    # Print final comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Ground Truth: {results['gt_equation']}")
    print(f"Discovered:   {results['best_equation']}")
    print(f"\nGT Parameters: {results['gt_parameters']}")
    if system.buffer.get_best():
        print(f"Fitted Parameters: {system.buffer.get_best().parameters}")
    print("="*70)


if __name__ == "__main__":
    main()
