#!/usr/bin/env python3
"""
PDE Discovery - AutoGen v0.4 Implementation

Features:
- AutoGen v0.4 (autogen-agentchat) with AssistantAgent
- Tool use with direct tool execution in AssistantAgent
- TensorBoard logging for metrics tracking
- Experience buffer with in-context learning
- Asynchronous event-driven architecture
"""

import argparse
import json
import time
from pathlib import Path
from typing import Annotated
import numpy as np
import asyncio

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# AutoGen v0.4
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily


from bench.pde_datamodule import ChemotaxisDataModule
from bench.pde_solver import PDESolver, PDEConfig
from bench.pde_visualization import PDEVisualizer
from bench.pde_experience_buffer import PDEExperienceBuffer


class PDEDiscoveryAutogenV04:
    """
    PDE Discovery with AutoGen v0.4 AssistantAgent with tools
    """

    def __init__(
        self,
        api_base: str = "http://localhost:10005/v1",
        api_model: str = "/mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct",
        max_iterations: int = 8000,
        samples_per_prompt: int = 4,
        convergence_threshold: float = 0.98,
        plateau_patience: int = 100,
        output_dir: str = "./logs/pde_discovery_autogen_v04",
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

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        print(f"✓ TensorBoard logging to: {self.output_dir / 'tensorboard'}")

        # State for tool calls
        self.current_problem = None
        self.best_score = -float('inf')
        self.best_equation = None
        self.plateau_counter = 0
        self.iteration = 0

        # Setup
        self._setup_model_client()

    def _setup_model_client(self):
        """Setup OpenAI-compatible model client for AutoGen v0.4"""

        # Create model client with custom base_url for vLLM
        self.model_client = OpenAIChatCompletionClient(
            model=self.api_model,
            base_url=self.api_base,
            api_key="EMPTY",  # vLLM doesn't require real API key
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.UNKNOWN,
                "structured_output": False,
            },
        )
        print(f"✓ Model client initialized: {self.api_model}")

    def evaluate_pde_tool(
        self,
        equation: Annotated[str, "The PDE equation string to evaluate, e.g., 'α·Δg - β·∇·(g∇(ln S))'"]
    ) -> str:
        """
        Tool function: Evaluate PDE candidate

        This tool fits parameters and computes metrics for a given PDE equation.
        Returns a JSON string with results.
        """
        try:
            if self.current_problem is None:
                return json.dumps({'success': False, 'error': 'No problem loaded', 'score': 0.0})

            problem = self.current_problem

            param_bounds = {
                'α': (0.01, 3.0),
                'β': (0.01, 3.0),
                'γ': (0.001, 1.0),
                'K': (0.5, 10.0)
            }

            # Fit parameters
            fitted_params, loss = self.solver.fit_pde_parameters(
                equation, problem.g_init, problem.S, problem.g_observed,
                param_bounds=param_bounds
            )

            # Evaluate with fitted parameters
            predicted, info = self.solver.evaluate_pde(
                equation, problem.g_init, problem.S, fitted_params,
                num_steps=problem.g_observed.shape[2]
            )

            # Compute metrics
            mse = float(self.solver.compute_spatiotemporal_loss(predicted, problem.g_observed, 'mse'))
            r2 = float(self.solver.compute_spatiotemporal_loss(predicted, problem.g_observed, 'r2'))
            nmse = float(self.solver.compute_spatiotemporal_loss(predicted, problem.g_observed, 'nmse'))

            obs_mass = np.sum(problem.g_observed, axis=(0, 1))
            pred_mass = np.sum(predicted, axis=(0, 1))
            mass_error = float(np.abs(pred_mass[-1] - obs_mass[-1]) / obs_mass[-1] * 100)

            # Composite score
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
                self.plateau_counter = 0

                if self.writer:
                    self.writer.add_scalar('best/score', score, self.iteration)
                    self.writer.add_scalar('best/r2', r2, self.iteration)

                # Save visualization every 200 iterations
                if self.iteration % 200 == 0:
                    viz_path = self.output_dir / f"best_iter_{self.iteration:06d}.png"
                    self.visualizer.create_critique_visualization(
                        problem.g_observed, predicted,
                        equation, {'mse': mse, 'r2': r2, 'nmse': nmse, 'mass_error': mass_error},
                        save_path=str(viz_path)
                    )
                    if self.writer:
                        try:
                            from PIL import Image
                            import torchvision.transforms as transforms
                            img = Image.open(viz_path)
                            img_tensor = transforms.ToTensor()(img)
                            self.writer.add_image('visualizations/best', img_tensor, self.iteration)
                        except:
                            pass

                print(f"\n🎯 Iter {self.iteration}: NEW BEST! Score={score:.4f}, R²={r2:.4f}")
                print(f"   Equation: {equation[:100]}...")
            else:
                self.plateau_counter += 1

            result = {
                'success': True,
                'score': float(score),
                'r2': float(r2),
                'mse': float(mse),
                'mass_error': float(mass_error),
                'fitted_params': {k: float(v) for k, v in fitted_params.items()},
                'message': f"✓ Evaluated! Score: {score:.4f}, R²: {r2:.4f}, MSE: {mse:.6f}, Mass Error: {mass_error:.2f}%"
            }

            return json.dumps(result)

        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'score': 0.0,
                'message': f"✗ Evaluation failed: {str(e)}"
            }
            return json.dumps(error_result)

    async def discover(self, problem, verbose: bool = True):
        """Main discovery loop with AutoGen v0.4"""

        self.current_problem = problem
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
        }

        start_time = time.time()

        if verbose:
            print("\n" + "="*70)
            print("PDE DISCOVERY - AUTOGEN V0.4")
            print("="*70)
            print(f"Dataset: {data_summary['shape']}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Ground Truth: {problem.gt_equation}")
            print(f"Mass change: {data_summary['mass_change_pct']:.2f}%")
            print("="*70)

        # Create AssistantAgent with evaluate_pde_tool
        assistant = AssistantAgent(
            name="PDE_Generator",
            model_client=self.model_client,
            tools=[self.evaluate_pde_tool],
            reflect_on_tool_use=True,  # Reflect on tool results
            system_message=f"""You are an expert in mathematical biology and PDE modeling.
Your task is to discover PDEs for chemotaxis phenomena from spatiotemporal data.

DATA SUMMARY:
- Grid: {data_summary['shape']} (Height × Width × Time)
- Cell density: [{data_summary['g_min']:.4f}, {data_summary['g_max']:.4f}]
- Mass change: {data_summary['mass_change_pct']:.2f}%
- Attractant S: [{data_summary['S_min']:.4f}, {data_summary['S_max']:.4f}]

AVAILABLE OPERATORS:
- ∇ (gradient)
- ∇· (divergence)
- Δ (Laplacian = ∇²)
- ∂/∂t (time derivative)
- ln (natural logarithm)

CHEMOTAXIS TERMS TO CONSIDER:
1. Diffusion: α·Δg (random motion)
2. Chemotaxis: -β·∇·(g∇(ln S)) or -β·∇·(g∇S) (directed motion)
3. Growth: γ·g(1-g/K) (logistic growth)

OUTPUT FORMAT:
Propose PDE equations in the form: ∂g/∂t = [right-hand side]

INSTRUCTIONS:
1. Generate {self.samples_per_prompt} diverse PDE candidates
2. For EACH candidate, call evaluate_pde(equation="your_pde") tool
3. Analyze the results (score, R², mass error)
4. Propose refined candidates based on previous results
5. Try different operator combinations
6. Explore parameter-free forms first, parameters will be fitted automatically

Generate creative and diverse PDEs!""",
        )

        # Discovery loop
        for iteration in range(1, self.max_iterations + 1):
            self.iteration = iteration
            iter_start = time.time()

            # Get top-5 experience context
            experience_context = self.buffer.format_for_prompt(k=3, include_visual=True)

            # Create prompt with context
            if experience_context:
                prompt_with_context = f"""Generate {self.samples_per_prompt} NEW PDE candidates and evaluate them.

PREVIOUS TOP RESULTS:
{experience_context}

Learn from these results and propose IMPROVED or DIFFERENT candidates.
Call evaluate_pde for each candidate."""
            else:
                prompt_with_context = f"Generate {self.samples_per_prompt} diverse PDE candidates and evaluate each using evaluate_pde tool."

            # Run assistant
            try:
                cancellation_token = CancellationToken()
                response = await assistant.on_messages(
                    [TextMessage(content=prompt_with_context, source="user")],
                    cancellation_token
                )

                if verbose and iteration % 2 == 0:
                    print(f"\n[Iter {iteration}] Agent response: {response.chat_message.content[:200]}...")

            except Exception as e:
                if verbose:
                    print(f"Iteration {iteration} failed: {e}")
                continue

            # Log iteration time
            iter_time = time.time() - iter_start
            if self.writer:
                self.writer.add_scalar('performance/iteration_time', iter_time, iteration)
                self.writer.add_scalar('performance/buffer_size', len(self.buffer), iteration)
                self.writer.add_scalar('performance/plateau_counter', self.plateau_counter, iteration)

            # Progress reporting
            if verbose and iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"\n{'='*70}")
                print(f"Progress: Iteration {iteration}/{self.max_iterations}")
                print(f"Best Score: {self.best_score:.4f} | Plateau: {self.plateau_counter}/{self.plateau_patience}")
                print(f"Buffer Size: {len(self.buffer)} PDEs | Elapsed: {elapsed:.1f}s")
                if self.best_equation:
                    print(f"Best Equation: {self.best_equation[:80]}...")
                print(f"{'='*70}")

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

        # Close model client
        await self.model_client.close()

        if verbose:
            print("\n" + "="*70)
            print("DISCOVERY COMPLETE")
            print("="*70)
            print(f"Best Equation: {self.best_equation}")
            print(f"Best Score: {self.best_score:.4f}")
            print(f"Total Iterations: {iteration}")
            print(f"Total Time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
            print(f"Results saved to: {results_path}")
            if self.writer:
                print(f"\nTensorBoard: tensorboard --logdir {self.output_dir / 'tensorboard'} --port 6006")
            print("="*70)

        return results


async def main_async():
    parser = argparse.ArgumentParser(description="PDE Discovery - AutoGen v0.4")
    parser.add_argument('--dataset', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--api_base', type=str, default='http://localhost:10005/v1', help='vLLM API URL')
    parser.add_argument('--api_model', type=str, default='/mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct', help='Model path')
    parser.add_argument('--max_iterations', type=int, default=8000, help='Max iterations')
    parser.add_argument('--samples_per_prompt', type=int, default=4, help='Samples per prompt')
    parser.add_argument('--output_dir', type=str, default='./logs/pde_discovery_autogen_v04', help='Output directory')
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dm = ChemotaxisDataModule(data_source="hdf5", data_path=args.dataset)
    problems = dm.load()
    problem = list(problems.values())[0]

    print(f"✓ Loaded: {problem.g_observed.shape}")
    print(f"  Ground Truth: {problem.gt_equation}")

    # Run discovery
    system = PDEDiscoveryAutogenV04(
        api_base=args.api_base,
        api_model=args.api_model,
        max_iterations=args.max_iterations,
        samples_per_prompt=args.samples_per_prompt,
        output_dir=args.output_dir
    )

    results = await system.discover(problem, verbose=True)

    # Final comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Ground Truth: {results['gt_equation']}")
    print(f"Discovered:   {results['best_equation']}")
    print(f"\nGT Parameters: {results['gt_parameters']}")
    if system.buffer.get_best():
        print(f"Fitted Params: {system.buffer.get_best().parameters}")
    print("="*70)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
