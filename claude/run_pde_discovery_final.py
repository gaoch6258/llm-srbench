#!/usr/bin/env python3
"""
PDE Discovery - Final Version with TensorBoard and AutoGen

Features:
- Full AutoGen dual-agent system
- TensorBoard logging for metrics tracking
- Visual critic with image analysis
- Experience buffer with in-context learning
- LLM-SR style iterative refinement
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False

# AutoGen
try:
    from autogen import ConversableAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    print("Warning: AutoGen not available. Install with: pip install pyautogen")
    AUTOGEN_AVAILABLE = False

from bench.pde_datamodule import ChemotaxisDataModule
from bench.pde_solver import PDESolver, PDEConfig
from bench.pde_visualization import PDEVisualizer
from bench.pde_experience_buffer import PDEExperienceBuffer
from bench.pde_prompts import (
    create_generator_prompt,
    create_critic_prompt,
    extract_equation_from_response,
    extract_reasoning_from_response,
    extract_scores_from_critique,
    extract_suggestions_from_critique
)


class PDEDiscoveryWithTensorBoard:
    """
    PDE Discovery with full AutoGen and TensorBoard logging
    """

    def __init__(
        self,
        api_base: str = "http://localhost:10005/v1",
        api_model: str = "/mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct",
        max_iterations: int = 8000,
        samples_per_prompt: int = 4,
        convergence_threshold: float = 0.98,
        plateau_patience: int = 100,
        output_dir: str = "./logs/pde_discovery",
        use_autogen: bool = True,
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
        self.use_autogen = use_autogen and AUTOGEN_AVAILABLE

        # Components
        self.solver = PDESolver(solver_config or PDEConfig())
        self.visualizer = PDEVisualizer()
        self.buffer = PDEExperienceBuffer(max_size=200)

        # TensorBoard
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
            print(f"✓ TensorBoard logging to: {self.output_dir / 'tensorboard'}")
        else:
            self.writer = None

        # State
        self.best_score = -float('inf')
        self.best_equation = None
        self.plateau_counter = 0
        self.iteration = 0

        # Setup
        if self.use_autogen:
            self._setup_autogen_agents()
        else:
            self._setup_llm_client()

    def _setup_llm_client(self):
        """Setup OpenAI-compatible client"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key="EMPTY", base_url=self.api_base)
            print(f"✓ Connected to LLM at {self.api_base}")
        except Exception as e:
            print(f"Warning: Could not connect: {e}")
            self.client = None

    def _setup_autogen_agents(self):
        """Setup AutoGen agents"""
        if not AUTOGEN_AVAILABLE:
            print("AutoGen not available, falling back to direct LLM")
            self.use_autogen = False
            self._setup_llm_client()
            return

        llm_config = {
            "model": self.api_model,
            "api_key": "EMPTY",
            "base_url": self.api_base,
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        # Generator agent
        self.generator_agent = ConversableAgent(
            name="PDEGenerator",
            system_message="""You are an expert in mathematical biology and PDE modeling.
Generate novel PDE candidates for chemotaxis. Use operators: ∇, ∇·, Δ, ∂/∂t.
Always output in structured format with <equation>, <reasoning>, <parameters> tags.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )

        # Critic agent (for text-only analysis, vision requires special handling)
        self.critic_agent = ConversableAgent(
            name="PDECritic",
            system_message="""You are an expert in PDE analysis and chemotaxis.
Analyze PDE solutions and provide scores (0-10) and suggestions.
Output in structured format with <scores>, <analysis>, <suggestions> tags.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )

        print(f"✓ AutoGen agents initialized")

    def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Call LLM (AutoGen or direct)"""
        if self.use_autogen:
            try:
                response = self.generator_agent.generate_reply(
                    messages=[{"role": "user", "content": prompt}]
                )
                if isinstance(response, dict):
                    return response.get('content', '')
                return str(response)
            except Exception as e:
                print(f"AutoGen call failed: {e}")
                return ""
        else:
            if self.client is None:
                return ""
            try:
                response = self.client.chat.completions.create(
                    model=self.api_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1024
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"LLM call failed: {e}")
                return ""

    def _log_to_tensorboard(self, tag: str, value: float, step: int):
        """Log scalar to TensorBoard"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def _log_image_to_tensorboard(self, tag: str, image_path: str, step: int):
        """Log image to TensorBoard"""
        if self.writer:
            try:
                from PIL import Image
                import torch
                import torchvision.transforms as transforms

                img = Image.open(image_path)
                transform = transforms.ToTensor()
                img_tensor = transform(img)
                self.writer.add_image(tag, img_tensor, step)
            except Exception as e:
                print(f"Could not log image: {e}")

    def _generate_candidates(self, problem, data_summary, num_candidates: int = 4) -> list:
        """Generate PDE candidates"""
        experience_context = self.buffer.format_for_prompt(k=5, include_visual=False)
        problem_desc = f"Discover PDE for chemotaxis from {data_summary['shape']} data"
        prompt = create_generator_prompt(problem_desc, data_summary, experience_context, self.iteration)

        candidates = []
        for i in range(num_candidates):
            response = self._call_llm(prompt, temperature=0.7 + i * 0.1)
            equation = extract_equation_from_response(response)
            reasoning = extract_reasoning_from_response(response)

            if equation:
                candidates.append({'equation': equation, 'reasoning': reasoning})

        return candidates

    def _evaluate_candidate(self, equation: str, problem) -> dict:
        """Evaluate PDE candidate"""
        try:
            param_bounds = {
                'α': (0.01, 3.0),
                'β': (0.01, 3.0),
                'γ': (0.001, 1.0),
                'K': (0.5, 10.0)
            }

            fitted_params, loss = self.solver.fit_pde_parameters(
                equation, problem.g_init, problem.S, problem.g_observed,
                param_bounds=param_bounds
            )

            predicted, info = self.solver.evaluate_pde(
                equation, problem.g_init, problem.S, fitted_params,
                num_steps=problem.g_observed.shape[2]
            )

            mse = float(self.solver.compute_spatiotemporal_loss(predicted, problem.g_observed, 'mse'))
            r2 = float(self.solver.compute_spatiotemporal_loss(predicted, problem.g_observed, 'r2'))
            nmse = float(self.solver.compute_spatiotemporal_loss(predicted, problem.g_observed, 'nmse'))

            obs_mass = np.sum(problem.g_observed, axis=(0, 1))
            pred_mass = np.sum(predicted, axis=(0, 1))
            mass_error = float(np.abs(pred_mass[-1] - obs_mass[-1]) / obs_mass[-1] * 100)

            score = r2 * 10 * (1 - min(mass_error / 100, 0.5))

            return {
                'success': True,
                'predicted': predicted,
                'fitted_params': fitted_params,
                'metrics': {'mse': mse, 'r2': r2, 'nmse': nmse, 'mass_error': mass_error},
                'score': score
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'score': -1.0}

    def discover(self, problem, verbose: bool = True):
        """Main discovery loop with TensorBoard logging"""
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
            print(f"AutoGen: {'ENABLED' if self.use_autogen else 'DISABLED'}")
            print(f"TensorBoard: {'ENABLED' if self.writer else 'DISABLED'}")
            print(f"Dataset: {data_summary['shape']}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"GT: {problem.gt_equation}")
            print("="*70)

        # Discovery loop
        for iteration in range(1, self.max_iterations + 1):
            self.iteration = iteration
            iter_start = time.time()

            # Generate and evaluate candidates
            candidates = self._generate_candidates(problem, data_summary, self.samples_per_prompt)

            for cand in candidates:
                equation = cand['equation']
                reasoning = cand['reasoning']

                if not equation:
                    continue

                result = self._evaluate_candidate(equation, problem)

                if result['success']:
                    score = result['score']
                    metrics = result['metrics']

                    # Log to TensorBoard
                    self._log_to_tensorboard('metrics/score', score, iteration)
                    self._log_to_tensorboard('metrics/r2', metrics['r2'], iteration)
                    self._log_to_tensorboard('metrics/mse', metrics['mse'], iteration)
                    self._log_to_tensorboard('metrics/mass_error', metrics['mass_error'], iteration)

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

                        self._log_to_tensorboard('best/score', score, iteration)
                        self._log_to_tensorboard('best/r2', metrics['r2'], iteration)

                        if verbose and iteration % 10 == 0:
                            print(f"\nIter {iteration}: NEW BEST! Score={score:.4f}, R²={metrics['r2']:.4f}")
                            print(f"  {equation[:80]}...")

                        # Save visualization every 200 iterations
                        if iteration % 200 == 0:
                            viz_path = self.output_dir / f"best_iter_{iteration:06d}.png"
                            self.visualizer.create_critique_visualization(
                                problem.g_observed, result['predicted'],
                                equation, metrics, save_path=str(viz_path)
                            )
                            self._log_image_to_tensorboard('visualizations/best', str(viz_path), iteration)

                    else:
                        self.plateau_counter += 1

            # Log iteration time
            iter_time = time.time() - iter_start
            self._log_to_tensorboard('performance/iteration_time', iter_time, iteration)
            self._log_to_tensorboard('performance/buffer_size', len(self.buffer), iteration)
            self._log_to_tensorboard('performance/plateau_counter', self.plateau_counter, iteration)

            # Progress
            if verbose and iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"\n--- Iter {iteration}/{self.max_iterations} ---")
                print(f"Best: {self.best_score:.4f} | Plateau: {self.plateau_counter}/{self.plateau_patience}")
                print(f"Buffer: {len(self.buffer)} | Time: {elapsed:.1f}s")

            # Convergence check
            if self.best_score >= self.convergence_threshold * 10:
                if verbose:
                    print(f"\n✓ CONVERGED at iteration {iteration}!")
                break

            if self.plateau_counter >= self.plateau_patience:
                if verbose:
                    print(f"\n⚠ Plateau at iteration {iteration}")
                break

        total_time = time.time() - start_time

        # Save final results
        results = {
            'success': self.best_equation is not None,
            'best_equation': self.best_equation,
            'best_score': float(self.best_score),
            'total_iterations': iteration,
            'total_time': total_time,
            'buffer_stats': self.buffer.get_statistics(),
            'gt_equation': problem.gt_equation,
            'gt_parameters': {k: v for k, v in problem.metadata.items() if k.endswith('_true')}
        }

        results_path = self.output_dir / "discovery_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        buffer_path = self.output_dir / "experience_buffer.json"
        self.buffer.save(str(buffer_path))

        if self.writer:
            self.writer.close()

        if verbose:
            print("\n" + "="*70)
            print("DISCOVERY COMPLETE")
            print("="*70)
            print(f"Best: {self.best_equation}")
            print(f"Score: {self.best_score:.4f}")
            print(f"Iterations: {iteration} | Time: {total_time:.1f}s")
            print(f"Results: {results_path}")
            if self.writer:
                print(f"TensorBoard: tensorboard --logdir {self.output_dir / 'tensorboard'}")
            print("="*70)

        return results


def main():
    parser = argparse.ArgumentParser(description="PDE Discovery - Final Version")
    parser.add_argument('--dataset', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--api_base', type=str, default='http://localhost:10005/v1', help='vLLM API URL')
    parser.add_argument('--api_model', type=str, default='/mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct', help='Model path')
    parser.add_argument('--max_iterations', type=int, default=8000, help='Max iterations')
    parser.add_argument('--samples_per_prompt', type=int, default=4, help='Samples per prompt')
    parser.add_argument('--use_autogen', action='store_true', help='Use AutoGen agents')
    parser.add_argument('--output_dir', type=str, default='./logs/pde_discovery_final', help='Output dir')
    args = parser.parse_args()

    # Load dataset
    print(f"Loading: {args.dataset}")
    dm = ChemotaxisDataModule(data_source="hdf5", data_path=args.dataset)
    problems = dm.load()
    problem = list(problems.values())[0]

    print(f"✓ Loaded: {problem.g_observed.shape}")
    print(f"  GT: {problem.gt_equation}")

    # Run discovery
    system = PDEDiscoveryWithTensorBoard(
        api_base=args.api_base,
        api_model=args.api_model,
        max_iterations=args.max_iterations,
        samples_per_prompt=args.samples_per_prompt,
        use_autogen=args.use_autogen,
        output_dir=args.output_dir
    )

    results = system.discover(problem, verbose=True)

    # Final comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"GT: {results['gt_equation']}")
    print(f"Discovered: {results['best_equation']}")
    print(f"\nGT Params: {results['gt_parameters']}")
    if system.buffer.get_best():
        print(f"Fitted: {system.buffer.get_best().parameters}")
    print("="*70)


if __name__ == "__main__":
    main()
