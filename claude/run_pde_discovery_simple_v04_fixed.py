#!/usr/bin/env python3
"""
PDE Discovery - AutoGen v0.4 WITHOUT Tool Calling + Context Management

Fixed: Clears agent state regularly to prevent context overflow
"""

import argparse
import json
import time
import re
from pathlib import Path
import numpy as np
import asyncio

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# AutoGen v0.4
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_core import CancellationToken
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
except ImportError:
    print("Warning: AutoGen v0.4 not available")
    AUTOGEN_AVAILABLE = False

from bench.pde_datamodule import ChemotaxisDataModule
from bench.pde_solver import PDESolver, PDEConfig
from bench.pde_visualization import PDEVisualizer
from bench.pde_experience_buffer import PDEExperienceBuffer


def extract_equations_from_text(text: str) -> list:
    """Extract PDE equations from agent response"""
    equations = []

    # Pattern 1: Equations in code blocks
    code_blocks = re.findall(r'```(?:pde|math)?\s*\n(.*?)\n```', text, re.DOTALL)
    for block in code_blocks:
        lines = block.strip().split('\n')
        for line in lines:
            if '∂g/∂t' in line or 'dg/dt' in line:
                eq = line.split('=', 1)[-1].strip()
                if eq and len(eq) > 5:
                    equations.append(eq)

    # Pattern 2: Equations after "∂g/∂t ="
    matches = re.findall(r'∂g/∂t\s*=\s*([^\n]+)', text)
    equations.extend([m.strip() for m in matches if len(m.strip()) > 5])

    # Pattern 3: Numbered equations
    matches = re.findall(r'\d+\.\s*([^=]*=\s*[^\n]+)', text)
    for match in matches:
        if '∂g/∂t' in match or any(op in match for op in ['Δg', '∇', 'α', 'β']):
            eq = match.split('=', 1)[-1].strip()
            if eq and len(eq) > 5:
                equations.append(eq)

    # Remove duplicates and clean
    unique_eqs = []
    seen = set()
    for eq in equations:
        eq_clean = eq.strip('.,;:')
        if eq_clean and eq_clean not in seen and len(eq_clean) > 5:
            seen.add(eq_clean)
            unique_eqs.append(eq_clean)

    return unique_eqs[:10]  # Limit to top 10


class PDEDiscoverySimpleV04Fixed:
    """PDE Discovery with AutoGen v0.4 + Context Management"""

    def __init__(
        self,
        api_base: str = "http://localhost:10005/v1",
        api_model: str = "/mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct",
        max_iterations: int = 8000,
        samples_per_prompt: int = 4,
        convergence_threshold: float = 0.98,
        plateau_patience: int = 100,
        output_dir: str = "./logs/pde_discovery_simple_v04",
        solver_config: PDEConfig = None,
        reset_interval: int = 50  # Reset agent every N iterations to clear context
    ):
        self.api_base = api_base
        self.api_model = api_model
        self.max_iterations = max_iterations
        self.samples_per_prompt = samples_per_prompt
        self.convergence_threshold = convergence_threshold
        self.plateau_patience = plateau_patience
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reset_interval = reset_interval

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
        self.best_params = None  # Store fitted parameters
        self.best_metrics = None  # Store best metrics
        self.plateau_counter = 0
        self.iteration = 0

        if not AUTOGEN_AVAILABLE:
            raise RuntimeError("AutoGen v0.4 required")

        self._setup_model_client()

    def _setup_model_client(self):
        """Setup model client WITHOUT tool calling"""
        self.model_client = OpenAIChatCompletionClient(
            model=self.api_model,
            base_url=self.api_base,
            api_key="EMPTY",
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "family": "unknown",
            },
        )
        print(f"✓ Model client initialized: {self.api_model}")

    def _create_assistant(self, system_message: str) -> AssistantAgent:
        """Create a fresh AssistantAgent"""
        return AssistantAgent(
            name="PDE_Generator",
            model_client=self.model_client,
            system_message=system_message,
        )

    def evaluate_pde(self, equation: str, problem) -> dict:
        """Evaluate PDE candidate"""
        try:
            param_bounds = {
                'α': (0.01, 3.0),
                'β': (0.01, 3.0),
                'γ': (0.001, 1.0),
                'K': (0.5, 10.0)
            }

            fitted_params, _ = self.solver.fit_pde_parameters(
                equation, problem.g_init, problem.S, problem.g_observed,
                param_bounds=param_bounds
            )

            predicted, _ = self.solver.evaluate_pde(
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

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar('metrics/score', score, self.iteration)
                self.writer.add_scalar('metrics/r2', r2, self.iteration)
                self.writer.add_scalar('metrics/mse', mse, self.iteration)
                self.writer.add_scalar('metrics/mass_error', mass_error, self.iteration)

            # Add to buffer
            self.buffer.add(
                equation=equation,
                score=score,
                metrics={'mse': mse, 'r2': r2, 'nmse': nmse, 'mass_error': mass_error},
                visual_analysis="",
                reasoning="",
                suggestions="",
                parameters=fitted_params
            )

            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_equation = equation
                self.best_params = fitted_params  # Store fitted parameters
                self.best_metrics = {'mse': mse, 'r2': r2, 'nmse': nmse, 'mass_error': mass_error}
                self.plateau_counter = 0

                if self.writer:
                    self.writer.add_scalar('best/score', score, self.iteration)
                    self.writer.add_scalar('best/r2', r2, self.iteration)
                    self.writer.add_scalar('best/mse', mse, self.iteration)
                    self.writer.add_scalar('best/mass_error', mass_error, self.iteration)

                    # Log fitted parameters to TensorBoard
                    for param_name, param_value in fitted_params.items():
                        self.writer.add_scalar(f'best_params/{param_name}', param_value, self.iteration)

                # Save visualization MORE FREQUENTLY (every 50 iterations for new best)
                if self.iteration % 50 == 0 or score > self.best_score * 1.05:
                    viz_path = self.output_dir / f"best_iter_{self.iteration:06d}.png"
                    self.visualizer.create_critique_visualization(
                        problem.g_observed, predicted, equation,
                        {'mse': mse, 'r2': r2, 'nmse': nmse, 'mass_error': mass_error},
                        save_path=str(viz_path)
                    )

                # Print with FITTED PARAMETERS
                print(f"\n🎯 Iter {self.iteration}: NEW BEST! Score={score:.4f}, R²={r2:.4f}")
                print(f"   Equation: {equation[:80]}...")
                print(f"   Fitted Parameters: {', '.join([f'{k}={v:.4f}' for k, v in fitted_params.items()])}")
            else:
                self.plateau_counter += 1

            return {
                'success': True,
                'score': score,
                'r2': r2,
                'predicted': predicted
            }

        except Exception as e:
            return {'success': False, 'error': str(e), 'score': 0.0}

    async def discover(self, problem, verbose: bool = True):
        """Main discovery loop with context management"""

        H, W, T = problem.g_observed.shape
        data_summary = {
            'shape': f"({H}, {W}, {T})",
            'mass_change_pct': float((problem.g_observed[:, :, -1].sum() - problem.g_init.sum()) / problem.g_init.sum() * 100),
        }

        start_time = time.time()

        if verbose:
            print("\n" + "="*70)
            print("PDE DISCOVERY - AUTOGEN V0.4 (CONTEXT MANAGED)")
            print("="*70)
            print(f"Dataset: {data_summary['shape']}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Reset interval: {self.reset_interval} iterations")
            print(f"Ground Truth: {problem.gt_equation}")
            print(f"Mass change: {data_summary['mass_change_pct']:.2f}%")
            print("="*70)

        # Base system message
        base_system_message = f"""You are an expert in mathematical biology and PDE modeling.

Generate {self.samples_per_prompt} diverse PDE candidates for chemotaxis.

FORMAT: Always output equations as:
∂g/∂t = [right-hand side]

OPERATORS:
- Δg (Laplacian)
- ∇·(g∇S) or ∇·(g∇(ln S)) (chemotaxis)
- g(1-g/K) (growth)

EXAMPLES:
1. ∂g/∂t = α·Δg
2. ∂g/∂t = α·Δg - β·∇·(g∇(ln S))
3. ∂g/∂t = α·Δg - β·∇·(g∇(ln S)) + γ·g(1-g/K)

Generate {self.samples_per_prompt} NEW candidates now."""

        # Create initial assistant
        assistant = self._create_assistant(base_system_message)

        # Discovery loop
        for iteration in range(1, self.max_iterations + 1):
            self.iteration = iteration
            iter_start = time.time()

            # Reset assistant every N iterations to prevent context overflow
            if iteration % self.reset_interval == 0:
                if verbose:
                    print(f"\n♻️  Resetting agent at iteration {iteration} (clearing context)")
                # Create new assistant with fresh context
                assistant = self._create_assistant(base_system_message)

            # Get experience context (only top 3 to save tokens)
            experience_context = self.buffer.format_for_prompt(k=3, include_visual=False)

            # Create concise prompt
            if experience_context:
                prompt = f"""Generate {self.samples_per_prompt} NEW PDEs. Learn from top results:

{experience_context[:1000]}

Output {self.samples_per_prompt} equations as:
∂g/∂t = [expression]"""
            else:
                prompt = f"Generate {self.samples_per_prompt} diverse PDE equations. Format: ∂g/∂t = [expression]"

            # Get agent response
            try:
                cancellation_token = CancellationToken()
                response = await assistant.on_messages(
                    [TextMessage(content=prompt, source="user")],
                    cancellation_token
                )

                # Extract equations from response
                equations = extract_equations_from_text(response.chat_message.content)

                if verbose and iteration % 10 == 0:
                    print(f"\n[Iter {iteration}] Generated {len(equations)} equations")

                # Evaluate each equation
                for eq in equations[:self.samples_per_prompt]:
                    result = self.evaluate_pde(eq, problem)
                    if not result['success'] and verbose and iteration % 50 == 0:
                        print(f"  ✗ Failed: {eq[:60]}...")

            except Exception as e:
                if verbose:
                    print(f"\n❌ Iteration {iteration} failed: {e}")
                # Reset agent on error
                assistant = self._create_assistant(base_system_message)
                continue

            # Log performance
            iter_time = time.time() - iter_start
            if self.writer:
                self.writer.add_scalar('performance/iteration_time', iter_time, iteration)
                self.writer.add_scalar('performance/buffer_size', len(self.buffer), iteration)
                self.writer.add_scalar('performance/plateau_counter', self.plateau_counter, iteration)

            # Progress
            if verbose and iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"\n{'='*70}")
                print(f"Progress: {iteration}/{self.max_iterations}")
                print(f"Best: {self.best_score:.4f} | Plateau: {self.plateau_counter}/{self.plateau_patience}")
                print(f"Buffer: {len(self.buffer)} | Time: {elapsed:.1f}s")
                if self.best_equation:
                    print(f"Equation: {self.best_equation[:80]}...")
                print(f"{'='*70}")

            # Convergence
            if self.best_score >= self.convergence_threshold * 10:
                if verbose:
                    print(f"\n✓ CONVERGED at iteration {iteration}!")
                break

            if self.plateau_counter >= self.plateau_patience:
                if verbose:
                    print(f"\n⚠ Plateau at iteration {iteration}")
                break

        total_time = time.time() - start_time

        # Save results
        results = {
            'success': self.best_equation is not None,
            'best_equation': self.best_equation,
            'best_params': {k: float(v) for k, v in self.best_params.items()} if self.best_params else None,
            'best_metrics': self.best_metrics,
            'best_score': float(self.best_score),
            'total_iterations': iteration,
            'total_time': total_time,
            'buffer_stats': self.buffer.get_statistics(),
            'gt_equation': problem.gt_equation,
            'gt_parameters': {k: v for k, v in problem.metadata.items() if k.endswith('_true')}
        }

        with open(self.output_dir / "discovery_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        self.buffer.save(str(self.output_dir / "experience_buffer.json"))

        if self.writer:
            self.writer.close()

        await self.model_client.close()

        if verbose:
            print("\n" + "="*70)
            print("DISCOVERY COMPLETE")
            print("="*70)
            print(f"Symbolic Equation: {self.best_equation}")
            if self.best_params:
                print(f"Fitted Parameters: {', '.join([f'{k}={v:.4f}' for k, v in self.best_params.items()])}")
            print(f"Score: {self.best_score:.4f}")
            if self.best_metrics:
                print(f"Metrics: R²={self.best_metrics['r2']:.4f}, MSE={self.best_metrics['mse']:.6f}, Mass Error={self.best_metrics['mass_error']:.2f}%")
            print(f"Time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
            print("="*70)

        return results


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--api_base', type=str, default='http://localhost:10005/v1')
    parser.add_argument('--api_model', type=str, default='/mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct')
    parser.add_argument('--max_iterations', type=int, default=8000)
    parser.add_argument('--samples_per_prompt', type=int, default=4)
    parser.add_argument('--reset_interval', type=int, default=50, help='Reset agent every N iterations')
    parser.add_argument('--output_dir', type=str, default='./logs/pde_discovery_simple_v04')
    args = parser.parse_args()

    print(f"Loading: {args.dataset}")
    dm = ChemotaxisDataModule(data_source="hdf5", data_path=args.dataset)
    problems = dm.load()
    problem = list(problems.values())[0]

    print(f"✓ Loaded: {problem.g_observed.shape}")
    print(f"  GT: {problem.gt_equation}")

    system = PDEDiscoverySimpleV04Fixed(
        api_base=args.api_base,
        api_model=args.api_model,
        max_iterations=args.max_iterations,
        samples_per_prompt=args.samples_per_prompt,
        reset_interval=args.reset_interval,
        output_dir=args.output_dir
    )

    results = await system.discover(problem, verbose=True)

    print("\n" + "="*70)
    print("COMPARISON: GROUND TRUTH vs. DISCOVERED")
    print("="*70)
    print(f"\nGround Truth Equation:")
    print(f"  {results['gt_equation']}")
    print(f"\nGround Truth Parameters:")
    for k, v in results['gt_parameters'].items():
        print(f"  {k}: {v}")

    print(f"\nDiscovered Equation (Symbolic):")
    print(f"  {results['best_equation']}")
    print(f"\nDiscovered Parameters (Fitted):")
    if results['best_params']:
        for k, v in results['best_params'].items():
            print(f"  {k}: {v:.6f}")

    print(f"\nFinal Metrics:")
    if results['best_metrics']:
        print(f"  R²: {results['best_metrics']['r2']:.6f}")
        print(f"  MSE: {results['best_metrics']['mse']:.6e}")
        print(f"  NMSE: {results['best_metrics']['nmse']:.6f}")
        print(f"  Mass Error: {results['best_metrics']['mass_error']:.2f}%")
    print(f"  Score: {results['best_score']:.6f}")
    print("="*70)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
