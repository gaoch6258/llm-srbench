"""
Dual-Agent System for PDE Discovery using Qwen3-VL-8B-Instruct

Implements two collaborative agents:
1. Equation Generator: Proposes PDE candidates
2. Visual Critic: Analyzes visualizations and provides feedback

Uses AutoGen framework for multi-agent orchestration.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import tempfile
import json
import re

try:
    import autogen
    from autogen import ConversableAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    print("Warning: AutoGen not available. Install with: pip install pyautogen")

from bench.pde_solver import PDESolver, PDEConfig, detect_equation_type
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


class PDEDiscoverySystem:
    """
    Dual-agent system for discovering PDEs from spatiotemporal data

    Architecture:
    - Generator Agent: Proposes PDE candidates
    - Visual Critic Agent: Analyzes solution visualizations
    - Experience Buffer: Stores history for in-context learning
    - PDE Solver: Evaluates candidates numerically
    - Visualizer: Creates plots for critic
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_iterations: int = 20,
        convergence_threshold: float = 0.95,
        plateau_patience: int = 5,
        solver_config: Optional[PDEConfig] = None,
        buffer_size: int = 100,
        work_dir: Optional[str] = None
    ):
        """
        Initialize PDE discovery system

        Args:
            model_name: Model identifier for Qwen3-VL-8B
            api_base: API endpoint URL
            api_key: API key (use "EMPTY" for local vLLM)
            max_iterations: Maximum discovery iterations
            convergence_threshold: Score threshold for convergence
            plateau_patience: Iterations without improvement before stopping
            solver_config: PDE solver configuration
            buffer_size: Experience buffer size
            work_dir: Working directory for outputs
        """
        if not AUTOGEN_AVAILABLE:
            raise ImportError("AutoGen is required. Install with: pip install pyautogen")

        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.plateau_patience = plateau_patience

        # Initialize components
        self.solver = PDESolver(solver_config or PDEConfig())
        self.visualizer = PDEVisualizer()
        self.buffer = PDEExperienceBuffer(max_size=buffer_size)

        # Working directory
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Initialize agents
        self._setup_agents()

        # Discovery state
        self.best_score = -float('inf')
        self.best_equation = None
        self.plateau_counter = 0
        self.iteration = 0

    def _setup_agents(self):
        """Setup AutoGen agents with appropriate configurations"""

        # LLM configuration for both agents
        llm_config = {
            "model": self.model_name,
            "api_key": self.api_key,
            "base_url": self.api_base,
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        # Equation Generator Agent
        self.generator_agent = ConversableAgent(
            name="EquationGenerator",
            system_message="""You are an expert mathematical biologist specializing in PDE modeling.
Your role is to generate novel PDE candidates for neutrophil chemotaxis based on data,
physical principles, and feedback from previous attempts. Always output structured responses
with <equation>, <reasoning>, and <parameters> tags.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )

        # Visual Critic Agent (vision-enabled)
        critic_llm_config = llm_config.copy()
        critic_llm_config["vision"] = True  # Enable vision capability

        self.critic_agent = ConversableAgent(
            name="VisualCritic",
            system_message="""You are an expert in scientific visualization and PDE analysis.
Your role is to analyze visualizations of PDE solutions and provide detailed, constructive
critique. Evaluate spatial accuracy, temporal dynamics, physical plausibility, and error
characteristics. Always output structured responses with <scores>, <analysis>, <strengths>,
<weaknesses>, and <suggestions> tags.""",
            llm_config=critic_llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )

        # Coordinator agent (optional, for orchestration)
        self.coordinator_agent = ConversableAgent(
            name="Coordinator",
            system_message="""You coordinate the PDE discovery process between the Generator
and Critic. You relay information and ensure structured communication.""",
            llm_config=None,  # No LLM, just coordination logic
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
        )

    def discover(
        self,
        g_init: np.ndarray,
        S: np.ndarray,
        g_observed: np.ndarray,
        problem_description: str = "",
        verbose: bool = True
    ) -> Dict:
        """
        Main discovery loop

        Args:
            g_init: Initial cell density (H, W)
            S: Chemoattractant field (H, W) or (H, W, T)
            g_observed: Observed evolution (H, W, T)
            problem_description: Text description of problem
            verbose: Print progress

        Returns:
            Dictionary with discovery results
        """
        # Compute data summary
        data_summary = self._compute_data_summary(g_init, S, g_observed)

        # Discovery loop
        for iteration in range(1, self.max_iterations + 1):
            self.iteration = iteration

            if verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration}/{self.max_iterations}")
                print(f"{'='*60}")

            # Step 1: Generate PDE candidate
            equation_str, reasoning = self._generate_equation(
                problem_description, data_summary, verbose
            )

            if not equation_str:
                if verbose:
                    print("Failed to generate valid equation. Skipping iteration.")
                continue

            if verbose:
                print(f"\nProposed PDE: {equation_str}")
                print(f"Reasoning: {reasoning[:200]}...")

            # Step 2: Evaluate PDE numerically
            try:
                predicted, fitted_params, metrics = self._evaluate_equation(
                    equation_str, g_init, S, g_observed, verbose
                )
            except Exception as e:
                if verbose:
                    print(f"Evaluation failed: {e}")
                continue

            if verbose:
                print(f"Metrics: MSE={metrics['mse']:.6f}, R²={metrics['r2']:.4f}")

            # Step 3: Create visualization
            viz_path = self.work_dir / f"iteration_{iteration:03d}.png"
            viz_image = self.visualizer.create_critique_visualization(
                g_observed, predicted, equation_str, metrics, save_path=str(viz_path)
            )

            # Step 4: Get visual critique
            visual_analysis, score, suggestions = self._critique_visualization(
                str(viz_path), equation_str, metrics, iteration, verbose
            )

            if verbose:
                print(f"Visual Critic Score: {score:.2f}/10")
                print(f"Suggestions: {suggestions[:200]}...")

            # Step 5: Store in experience buffer
            self.buffer.add(
                equation=equation_str,
                score=score,
                metrics=metrics,
                visual_analysis=visual_analysis,
                reasoning=reasoning,
                suggestions=suggestions,
                parameters=fitted_params
            )

            # Step 6: Check for improvement
            if score > self.best_score:
                self.best_score = score
                self.best_equation = equation_str
                self.plateau_counter = 0

                if verbose:
                    print(f"✓ New best score! {score:.2f}")
            else:
                self.plateau_counter += 1

                if verbose:
                    print(f"No improvement ({self.plateau_counter}/{self.plateau_patience})")

            # Step 7: Check convergence
            if score >= self.convergence_threshold * 10:  # Score out of 10
                if verbose:
                    print(f"\n✓ Converged! Score {score:.2f} >= threshold {self.convergence_threshold * 10}")
                break

            if self.plateau_counter >= self.plateau_patience:
                if verbose:
                    print(f"\n⚠ Plateau detected. Stopping after {self.plateau_patience} iterations without improvement.")
                break

        # Prepare final results
        best_experience = self.buffer.get_best()

        results = {
            'success': best_experience is not None,
            'best_equation': self.best_equation,
            'best_score': self.best_score,
            'total_iterations': self.iteration,
            'best_experience': best_experience.to_dict() if best_experience else None,
            'buffer_stats': self.buffer.get_statistics(),
            'all_experiences': [exp.to_dict() for exp in self.buffer.experiences]
        }

        # Save buffer
        buffer_path = self.work_dir / "experience_buffer.json"
        self.buffer.save(str(buffer_path))
        results['buffer_path'] = str(buffer_path)

        return results

    def _compute_data_summary(self, g_init: np.ndarray, S: np.ndarray,
                             g_observed: np.ndarray) -> Dict:
        """Compute statistical summary of data"""
        H, W, T = g_observed.shape

        return {
            'shape': f"({H}, {W}, {T})",
            'H': H,
            'W': W,
            'T': T,
            'dx': self.solver.config.dx,
            'dy': self.solver.config.dy,
            'dt': self.solver.config.dt,
            'g_min': float(g_observed.min()),
            'g_max': float(g_observed.max()),
            'g_mean': float(g_observed.mean()),
            'g_std': float(g_observed.std()),
            'S_min': float(S.min()),
            'S_max': float(S.max()),
            'mass_initial': float(g_init.sum()),
            'mass_final': float(g_observed[:, :, -1].sum()),
            'mass_change_pct': float((g_observed[:, :, -1].sum() - g_init.sum()) / g_init.sum() * 100),
            'spatial_spread': 'N/A'  # Could compute actual spread metrics
        }

    def _generate_equation(self, problem_description: str, data_summary: Dict,
                          verbose: bool = False) -> Tuple[str, str]:
        """
        Generate PDE candidate using Generator agent

        Returns:
            (equation_str, reasoning)
        """
        # Get previous experiences for context
        previous_exp = self.buffer.format_for_prompt(k=5, include_visual=True)

        # Create prompt
        prompt = create_generator_prompt(
            problem_description, data_summary, previous_exp, self.iteration
        )

        # Query generator agent
        try:
            response = self.generator_agent.generate_reply(
                messages=[{"role": "user", "content": prompt}]
            )

            if isinstance(response, dict):
                response = response.get('content', '')

            # Extract structured outputs
            equation = extract_equation_from_response(response)
            reasoning = extract_reasoning_from_response(response)

            return equation, reasoning

        except Exception as e:
            if verbose:
                print(f"Generator failed: {e}")
            return "", ""

    def _evaluate_equation(self, equation_str: str, g_init: np.ndarray,
                          S: np.ndarray, g_observed: np.ndarray,
                          verbose: bool = False) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Evaluate PDE equation numerically

        Returns:
            (predicted, fitted_params, metrics)
        """
        T = g_observed.shape[2]

        # Parse and fit parameters
        fitted_params, loss = self.solver.fit_pde_parameters(
            equation_str,
            g_init,
            S,
            g_observed,
            param_bounds={'α': (0.01, 5.0), 'χ': (0.01, 10.0)},
            method='L-BFGS-B'
        )

        # Solve with fitted parameters
        predicted, info = self.solver.evaluate_pde(
            equation_str, g_init, S, fitted_params, num_steps=T
        )

        # Compute metrics
        metrics = {
            'mse': float(self.solver.compute_spatiotemporal_loss(predicted, g_observed, 'mse')),
            'rmse': float(self.solver.compute_spatiotemporal_loss(predicted, g_observed, 'rmse')),
            'nmse': float(self.solver.compute_spatiotemporal_loss(predicted, g_observed, 'nmse')),
            'r2': float(self.solver.compute_spatiotemporal_loss(predicted, g_observed, 'r2')),
        }

        # Mass conservation error
        obs_mass = np.sum(g_observed, axis=(0, 1))
        pred_mass = np.sum(predicted, axis=(0, 1))
        mass_error = np.abs(pred_mass[-1] - obs_mass[-1]) / obs_mass[-1] * 100
        metrics['mass_error'] = float(mass_error)

        return predicted, fitted_params, metrics

    def _critique_visualization(self, viz_path: str, equation_str: str,
                               metrics: Dict, iteration: int,
                               verbose: bool = False) -> Tuple[str, float, str]:
        """
        Get visual critique from Critic agent

        Returns:
            (visual_analysis, score, suggestions)
        """
        # Create critique prompt
        prompt = create_critic_prompt(equation_str, metrics, iteration)

        # Query critic agent with image
        try:
            # For vision models, include image in message
            message_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"file://{viz_path}"}}
            ]

            response = self.critic_agent.generate_reply(
                messages=[{"role": "user", "content": message_content}]
            )

            if isinstance(response, dict):
                response = response.get('content', '')

            # Extract structured outputs
            scores = extract_scores_from_critique(response)
            suggestions = extract_suggestions_from_critique(response)

            # Use average score if available, otherwise estimate from metrics
            if 'average' in scores:
                score = scores['average']
            elif 'overall' in scores:
                score = scores['overall']
            else:
                # Fallback: convert R² to 0-10 scale
                score = max(0, min(10, metrics.get('r2', 0) * 10))

            return response, score, suggestions

        except Exception as e:
            if verbose:
                print(f"Critic failed: {e}")

            # Fallback scoring based on metrics
            r2 = metrics.get('r2', 0)
            score = max(0, min(10, r2 * 10))

            return "Critique unavailable", score, "No suggestions"


# Simplified non-AutoGen version for testing
class SimplePDEDiscoverySystem:
    """
    Simplified version without AutoGen for testing purposes
    Uses direct API calls instead of agent framework
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        api_base: str = "http://localhost:8000/v1",
        max_iterations: int = 20,
        solver_config: Optional[PDEConfig] = None,
        work_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.api_base = api_base
        self.max_iterations = max_iterations

        self.solver = PDESolver(solver_config or PDEConfig())
        self.visualizer = PDEVisualizer()
        self.buffer = PDEExperienceBuffer(max_size=100)

        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.best_score = -float('inf')
        self.best_equation = None

    def discover(self, g_init: np.ndarray, S: np.ndarray, g_observed: np.ndarray,
                verbose: bool = True) -> Dict:
        """Simplified discovery using direct solver testing"""

        # Test a few candidate PDEs directly
        candidates = [
            ("∂g/∂t = α·Δg - ∇·(g∇(ln S))", {'α': 0.5}),  # Reference
            ("∂g/∂t = α·Δg", {'α': 0.5}),  # Pure diffusion
        ]

        results = []

        for equation, params in candidates:
            if verbose:
                print(f"\nTesting: {equation}")

            try:
                T = g_observed.shape[2]
                predicted, info = self.solver.evaluate_pde(
                    equation, g_init, S, params, num_steps=T
                )

                mse = self.solver.compute_spatiotemporal_loss(predicted, g_observed, 'mse')
                r2 = self.solver.compute_spatiotemporal_loss(predicted, g_observed, 'r2')

                metrics = {'mse': float(mse), 'r2': float(r2)}

                if verbose:
                    print(f"  MSE: {mse:.6f}, R²: {r2:.4f}")

                results.append({
                    'equation': equation,
                    'params': params,
                    'metrics': metrics,
                    'predicted': predicted
                })

            except Exception as e:
                if verbose:
                    print(f"  Failed: {e}")

        # Find best
        best = max(results, key=lambda x: x['metrics']['r2']) if results else None

        return {
            'success': best is not None,
            'best_equation': best['equation'] if best else None,
            'best_score': best['metrics']['r2'] * 10 if best else 0,
            'results': results
        }
