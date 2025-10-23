#!/usr/bin/env python3
"""
PDE Discovery - AutoGen v0.4 WITHOUT Tool Calling + Sliding Window Context

This version:
- Works with vLLM servers that don't have --enable-auto-tool-choice
- Uses sliding window to preserve recent conversation (vs. reset loses all context)
- Agent generates equations as text, we parse them
- Maintains conversation history for better learning
"""

import argparse
import json
import time
import re
from pathlib import Path
from typing import List
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
    from autogen_agentchat.messages import TextMessage, ChatMessage
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


class PDEDiscoverySimpleV04Sliding:
    """PDE Discovery with AutoGen v0.4 WITHOUT tool calling + sliding window context"""

    def __init__(
        self,
        api_base: str = "http://localhost:10005/v1",
        api_model: str = "/mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct",
        max_iterations: int = 8000,
        samples_per_prompt: int = 4,
        convergence_threshold: float = 0.98,
        plateau_patience: int = 100,
        output_dir: str = "./logs/pde_discovery_simple_v04_sliding",
        solver_config: PDEConfig = None,
        context_window_size: int = 12,  # Keep last N messages (conservative default)
        context_trim_interval: int = 5  # Trim every N iterations (aggressive trimming)
    ):
        self.api_base = api_base
        self.api_model = api_model
        self.max_iterations = max_iterations
        self.samples_per_prompt = samples_per_prompt
        self.convergence_threshold = convergence_threshold
        self.plateau_patience = plateau_patience
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Context management
        self.context_window_size = context_window_size
        self.context_trim_interval = context_trim_interval
        self.conversation_history: List[ChatMessage] = []

        # Token management: Estimate ~500 tokens per message on average
        # With 40,730 token limit, keep history under 30,000 tokens for safety
        self.max_message_length = 1000  # Truncate long messages to 1000 chars (~200 tokens)

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
                "family": "unknown",  # Required in AutoGen v0.4.7+
            },
        )
        print(f"✓ Model client initialized (no tool calling): {self.api_model}")
        print(f"✓ Context window: {self.context_window_size} messages (trimmed every {self.context_trim_interval} iterations)")
        print(f"✓ Token limit strategy: Keep only system message + last {self.context_window_size} messages")

    def _truncate_message_content(self, content: str, max_length: int = None) -> str:
        """Truncate message content to prevent token overflow"""
        if max_length is None:
            max_length = self.max_message_length

        if len(content) > max_length:
            return content[:max_length] + "... [truncated]"
        return content

    def _trim_conversation_history(self):
        """Keep only the most recent N messages to prevent context overflow"""
        if len(self.conversation_history) > self.context_window_size:
            old_count = len(self.conversation_history)
            # Keep only last N messages
            self.conversation_history = self.conversation_history[-self.context_window_size:]
            trimmed = old_count - len(self.conversation_history)
            print(f"   ✂️  Trimmed {trimmed} old messages, kept recent {len(self.conversation_history)}")

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar('context/history_size', len(self.conversation_history), self.iteration)
                self.writer.add_scalar('context/trimmed_count', trimmed, self.iteration)

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
                self.plateau_counter = 0

                if self.writer:
                    self.writer.add_scalar('best/score', score, self.iteration)
                    self.writer.add_scalar('best/r2', r2, self.iteration)

                # Save visualization
                if self.iteration % 200 == 0:
                    viz_path = self.output_dir / f"best_iter_{self.iteration:06d}.png"
                    self.visualizer.create_critique_visualization(
                        problem.g_observed, predicted, equation,
                        {'mse': mse, 'r2': r2, 'nmse': nmse, 'mass_error': mass_error},
                        save_path=str(viz_path)
                    )

                print(f"\n🎯 Iter {self.iteration}: NEW BEST! Score={score:.4f}, R²={r2:.4f}")
                print(f"   Equation: {equation[:100]}...")
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
        """Main discovery loop with sliding window context"""

        H, W, T = problem.g_observed.shape
        data_summary = {
            'shape': f"({H}, {W}, {T})",
            'mass_change_pct': float((problem.g_observed[:, :, -1].sum() - problem.g_init.sum()) / problem.g_init.sum() * 100),
        }

        start_time = time.time()

        if verbose:
            print("\n" + "="*70)
            print("PDE DISCOVERY - AUTOGEN V0.4 (NO TOOL CALLING, SLIDING WINDOW)")
            print("="*70)
            print(f"Dataset: {data_summary['shape']}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Context window: {self.context_window_size} messages")
            print(f"Trim interval: every {self.context_trim_interval} iterations")
            print(f"Ground Truth: {problem.gt_equation}")
            print(f"Mass change: {data_summary['mass_change_pct']:.2f}%")
            print("="*70)

        # System message
        system_message = f"""You are an expert in mathematical biology and PDE modeling.

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

        # Create AssistantAgent WITHOUT tools
        assistant = AssistantAgent(
            name="PDE_Generator",
            model_client=self.model_client,
            system_message=system_message,
        )

        # Discovery loop
        for iteration in range(1, self.max_iterations + 1):
            self.iteration = iteration
            iter_start = time.time()

            # Trim conversation history periodically
            if iteration > 1 and iteration % self.context_trim_interval == 0:
                if verbose:
                    print(f"\n🔄 [Iter {iteration}] Trimming conversation history...")
                self._trim_conversation_history()

            # Get experience context (top-3, truncated)
            experience_context = self.buffer.format_for_prompt(k=3, include_visual=False)

            # Create prompt
            if experience_context:
                # Truncate to 800 chars
                experience_context_short = experience_context[:800]
                prompt = f"""Generate {self.samples_per_prompt} NEW PDEs. Learn from top results:

{experience_context_short}

Output {self.samples_per_prompt} equations as:
∂g/∂t = [expression]"""
            else:
                prompt = f"Generate {self.samples_per_prompt} diverse PDE equations. Format: ∂g/∂t = [expression]"

            # Get agent response with conversation history
            try:
                cancellation_token = CancellationToken()

                # Build messages: conversation history + new prompt
                messages_to_send = list(self.conversation_history)
                messages_to_send.append(TextMessage(content=prompt, source="user"))

                response = await assistant.on_messages(
                    messages_to_send,
                    cancellation_token
                )

                # Update conversation history with TRUNCATED messages to save tokens
                prompt_truncated = self._truncate_message_content(prompt)
                self.conversation_history.append(TextMessage(content=prompt_truncated, source="user"))

                # Truncate response content if it's very long
                response_content = response.chat_message.content if hasattr(response.chat_message, 'content') else str(response.chat_message)
                response_truncated = self._truncate_message_content(response_content)
                self.conversation_history.append(TextMessage(content=response_truncated, source="assistant"))

                # Extract equations from response
                equations = extract_equations_from_text(response.chat_message.content)

                if verbose and iteration % 10 == 0:
                    print(f"\n[Iter {iteration}] Generated {len(equations)} equations | History: {len(self.conversation_history)} msgs")

                # Evaluate each equation
                for eq in equations[:self.samples_per_prompt]:
                    result = self.evaluate_pde(eq, problem)
                    if not result['success'] and verbose and iteration % 50 == 0:
                        print(f"  ✗ Failed: {eq[:60]}...")

            except Exception as e:
                if verbose:
                    print(f"\n❌ Iteration {iteration} failed: {e}")
                # On error, trim aggressively
                if len(self.conversation_history) > self.context_window_size // 2:
                    self.conversation_history = self.conversation_history[-(self.context_window_size // 2):]
                    print(f"   ⚠️  Aggressive trim to {len(self.conversation_history)} messages")
                continue

            # Log performance
            iter_time = time.time() - iter_start
            if self.writer:
                self.writer.add_scalar('performance/iteration_time', iter_time, iteration)
                self.writer.add_scalar('performance/buffer_size', len(self.buffer), iteration)
                self.writer.add_scalar('performance/plateau_counter', self.plateau_counter, iteration)
                self.writer.add_scalar('context/history_size', len(self.conversation_history), iteration)

            # Progress
            if verbose and iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"\n{'='*70}")
                print(f"Progress: {iteration}/{self.max_iterations}")
                print(f"Best: {self.best_score:.4f} | Plateau: {self.plateau_counter}/{self.plateau_patience}")
                print(f"Buffer: {len(self.buffer)} | History: {len(self.conversation_history)} msgs | Time: {elapsed:.1f}s")
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
            'best_score': float(self.best_score),
            'total_iterations': iteration,
            'total_time': total_time,
            'buffer_stats': self.buffer.get_statistics(),
            'context_config': {
                'window_size': self.context_window_size,
                'trim_interval': self.context_trim_interval,
                'final_history_size': len(self.conversation_history)
            },
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
            print(f"Best: {self.best_equation}")
            print(f"Score: {self.best_score:.4f}")
            print(f"Time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
            print(f"Final History: {len(self.conversation_history)} messages")
            print("="*70)

        return results


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--api_base', type=str, default='http://localhost:10005/v1')
    parser.add_argument('--api_model', type=str, default='/mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct')
    parser.add_argument('--max_iterations', type=int, default=8000)
    parser.add_argument('--samples_per_prompt', type=int, default=4)
    parser.add_argument('--context_window_size', type=int, default=20)
    parser.add_argument('--context_trim_interval', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./logs/pde_discovery_simple_v04_sliding')
    args = parser.parse_args()

    print(f"Loading: {args.dataset}")
    dm = ChemotaxisDataModule(data_source="hdf5", data_path=args.dataset)
    problems = dm.load()
    problem = list(problems.values())[0]

    print(f"✓ Loaded: {problem.g_observed.shape}")
    print(f"  GT: {problem.gt_equation}")

    system = PDEDiscoverySimpleV04Sliding(
        api_base=args.api_base,
        api_model=args.api_model,
        max_iterations=args.max_iterations,
        samples_per_prompt=args.samples_per_prompt,
        context_window_size=args.context_window_size,
        context_trim_interval=args.context_trim_interval,
        output_dir=args.output_dir
    )

    results = await system.discover(problem, verbose=True)

    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"GT: {results['gt_equation']}")
    print(f"Discovered: {results['best_equation']}")
    print("="*70)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
