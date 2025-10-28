#!/usr/bin/env python3
"""
PDE Discovery - AutoGen v0.4 Implementation (Refined)

Features:
- AutoGen v0.4 (autogen-agentchat) with AssistantAgent
- Direct tool execution with proper form validation
- TensorBoard logging for metrics tracking
- Experience buffer with in-context learning
- Asynchronous event-driven architecture
"""

import argparse
import json
import time
from pathlib import Path
from typing import Annotated, Optional, Dict, Any
import numpy as np
import asyncio
import base64
import urllib.request
from PIL import Image as PILImage
# TensorBoard
from torch.utils.tensorboard import SummaryWriter
import PIL
import requests
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image
from autogen_core.model_context import BufferedChatCompletionContext
# AutoGen v0.4
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily


from bench.pde_datamodule import ChemotaxisDataModule
from bench.pde_datamodule import ChemotaxisProblem
from bench.pde_solver import PDESolver, PDEConfig
from bench.pde_visualization import PDEVisualizer
from bench.pde_experience_buffer import PDEExperienceBuffer
from bench.pde_llmsr_solver import LLMSRPDESolver


# PDE Discovery Guide - NO PRIORS, LET LLM DISCOVER
SUPPORTED_FORMS_GUIDE = """
PDE DISCOVERY TASK:

You will discover spatiotemporal PDEs that govern the evolution of a density field g(x,y,t) given the chemoattractant field S(x,y).

INPUT DATA:
- g(x,y,t): Density field observations over space and time (H, W, T)
- S(x,y): Signal/chemoattractant field (given, static or time-varying) (H, W, T)
- Spatial grid: (H×W) with spacing dx, dy
- Time steps: T with step dt
- params: Array of parameters, e.g. [p0, p1, p2, ...]

RETURN:
- g_next: Updated density (H, W, T) at next time step (one-step roll out)


YOUR TASK:
Generate COMPLETE Python code for a PDE update function using scipy/numpy operators.

FUNCTION SIGNATURE (MANDATORY):
```python
def pde_update(g: np.ndarray, S: np.ndarray, dx: float, dy: float, dt: float, params: np.ndarray) -> np.ndarray:
    # At the beggining, you MUST give the symbolic form of this PDE in the commment (PDE FORM: ...), such as:
    # PDE FORM: ∂g/∂t = D·Δg - χ·∇·(g·∇S)
    import numpy as np
    import scipy.ndimage

    # Extract parameters
    p0 = params[0]  # e.g., diffusion strength
    p1 = params[1]  # e.g., advection/chemotaxis strength
    # ... define as many as needed

    # Use scipy/numpy operators (calculate only the necessary term using numpy or scipy):
    # Laplacian : laplacian_g = scipy.ndimage.laplace(g, axes=(0,1)) / (dx**2)
    # Derivatives: dg_dx = np.gradient(g, dx, axis=0), dg_dy = np.gradient(g, dy, axis=1)


    # Compute dg/dt according to your PDE
    # dg_dt = p0 * laplacian(g) + p1 * some_other_term + ...

    # Forward Euler: g_next = g + dt * dg_dt

    return g_next
```
SHAPE SAFETY RULES:

1. g and S are 3D: (H, W, T) - Height × Width × Time
2; ALL intermediate variables MUST be (H, W, T)
3. dg_dx = np.gradient(g, dx, axis=0) return only (H, W), should be stacked to get the gradient vector
4. Laplacian: scipy.ndimage.laplace(g) / (dx**2) → (H,W,T)
5. Element-wise ops (*, +, -) require matching shapes
6. Final g_next MUST be (H, W, T)

OPERATORS YOU CAN USE (via scipy.ndimage):
- Laplacian Δ: scipy.ndimage.laplace(g, axes=(0,1)) / (dx**2)
- Derivatives ∇: dg_dx = np.gradient(g, dx, axis=0), dg_dy = np.gradient(g, dy, axis=1)
- Divergence ∇·(flux): compute flux components then their gradients
- Nonlinear terms: g², g³, g·S, exp, ln, etc.
- ANY mathematical combination

EXAMPLES OF POSSIBLE PDE FORMS:
- Pure diffusion: ∂g/∂t = D·Δg
- With reaction: ∂g/∂t = D·Δg + r·g
- With advection: ∂g/∂t = D·Δg - v·∇g
- With chemotaxis: ∂g/∂t = D·Δg - χ·∇·(g·∇S)
- Nonlinear: ∂g/∂t = D·Δg + r·g·(1-g)
- Complex: ∂g/∂t = D·Δg - χ·∇·(g·∇ln(S)) + r·g·(1-g/K)

NUMBER OF PARAMETERS:
- You decide how many parameters your PDE needs
- Specify as num_params when calling evaluate_pde_tool
- More parameters = more flexible but harder to fit
- Typical range: 1-5 parameters

SCORING:
- High R² (>0.9): Good fit to observations
- Low mass error (<5%): Conserves total mass
- Visual assessment: Spatial patterns match

CRITICAL:
- At the beggining, you MUST give the symbolic form of this PDE in the commment (PDE FORM: ...)
- You can comment the intermediate variables' shape to avoid shape errors.
- You should pay attention on the shape, the term that multiply/add/subtract should have the same shape
- Generate COMPLETE pde_update() function code
- Use scipy/numpy for all spatial operators (Laplacian, gradients, etc.)
- Discover the governing equation from data
- Be creative with parameter combinations
- Try diverse PDE structures
"""


# AsyncLLMClient - moved to module level so it can be pickled
class AsyncLLMClient:
    """Async-compatible LLM client wrapper for LLMSR"""
    def __init__(self, model_client):
        self.model_client = model_client

    def generate(self, prompt):
        """Synchronous wrapper that handles async properly

        This can be called from:
        1. Subprocess workers (multiprocessing) - no event loop, create one
        2. Async context (main process) - use existing loop with asyncio.run_coroutine_threadsafe
        """
        import asyncio
        import threading
        from autogen_agentchat.messages import UserMessage
        from autogen_core import CancellationToken

        async def _async_generate():
            # Use proper message format for AutoGen v0.4
            response = await self.model_client.create(
                messages=[UserMessage(content=prompt, source="user")],
                cancellation_token=CancellationToken()
            )
            return response.content

        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in async context - need to run in thread pool to avoid blocking
            # Use run_in_executor to run the async function
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Create a new event loop in thread
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(_async_generate())
                        return result
                    finally:
                        new_loop.close()

                future = executor.submit(run_in_thread)
                result = future.result(timeout=120)  # 2 minute timeout
                return result

        except RuntimeError as e:
            if "no running event loop" in str(e).lower() or "no current event loop" in str(e).lower():
                # Good - no running loop. We're in subprocess. Create one.
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(_async_generate())
                    return result
                finally:
                    loop.close()
            else:
                # Some other RuntimeError - re-raise it
                raise


class PDEDiscoveryAutogenV04:
    """
    PDE Discovery with AutoGen v0.4 AssistantAgent with tools
    """

    def __init__(
        self,
        api_base: str = "http://localhost:10005/v1",
        api_model: str = "/mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct",
        critic_model: Optional[str] = None,
        max_iterations: int = 8000,
        samples_per_prompt: int = 4,
        convergence_threshold: float = 0.98,
        plateau_patience: int = 100,
        output_dir: str = "./logs/pde_discovery_autogen_v04",
        solver_config: PDEConfig = None
    ):
        self.api_base = api_base
        self.api_model = api_model
        self.critic_model = critic_model or api_model
        self.max_iterations = max_iterations
        self.samples_per_prompt = samples_per_prompt
        self.convergence_threshold = convergence_threshold
        self.plateau_patience = plateau_patience
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir = self.output_dir / "visuals"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Components
        config = solver_config or PDEConfig()
        self.solver = PDESolver(config)  # Keep for metrics computation

        # NEW: Pure LLMSR-style solver - LLM generates code directly
        # We'll create a simple LLM client wrapper
        self.llmsr_solver = None  # Will be initialized when we have model_client

        self.visualizer = PDEVisualizer()
        self.buffer = PDEExperienceBuffer(max_size=200)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        print(f"✓ TensorBoard logging to: {self.output_dir / 'tensorboard'}")

        # State for tool calls
        self.current_problem = None
        self.best_score = -float('inf')
        self.best_equation = None
        self.numerical_best_score = -float('inf')
        self.numerical_best_equation = None
        self.plateau_counter = 0
        self.iteration = 0
        self.eval_counter = 0  # counts tool calls to build unique filenames

        # Scoring weights
        self.visual_weight = 0.30  # weight on visual critic score (0-10)
        self.eval_max_steps = 200
        # Setup
        self._setup_model_client()

    def _setup_model_client(self):
        """Setup OpenAI-compatible model client for AutoGen v0.4"""

        # Create model client with custom base_url for vLLM
        self.model_client = OpenAIChatCompletionClient(
            model=self.api_model,
            base_url=self.api_base,
            api_key="EMPTY",  # vLLM doesn't require real API key
            max_tokens=30000,
            model_info={
                "vision": True,  # FIXED: Enable vision for visual critic
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.UNKNOWN,
                "structured_output": False,
            },
        )
        print(f"✓ Model client initialized: {self.api_model}")

        # Create LLMSR solver with async-compatible LLM client wrapper
        # Use module-level AsyncLLMClient (defined at top of file, can be pickled)
        llm_client = AsyncLLMClient(self.model_client)

        self.llmsr_solver = LLMSRPDESolver(
            llm_client=llm_client,
            dx=1.0,
            dy=1.0,
            dt=0.01,
            timeout=60  # Longer timeout for code generation
        )
        print(f"✓ LLMSR solver initialized with LLM code generation")

    # Deprecated HTTP critic removed; we use critic_agent in discover()

    def evaluate_pde(
        self,
        pde_code: Annotated[str, "Complete Python code for pde_update function"],
        num_params: Annotated[int, "Number of parameters in params array"] = 2
    ) -> str:
        """
        Tool function: Evaluate PDE candidate code directly

        The agent provides complete pde_update() function code.
        Parameters are fit to data automatically.
        """
        try:
            if self.current_problem is None:
                return json.dumps({'success': False, 'error': 'No problem loaded', 'score': 0.0})

            problem = self.current_problem

            # Parameter bounds - generic for any number of params
            param_bounds = [(1e-6, 1e2) for _ in range(num_params)]

            # Directly use the provided code (agent generates it)
            # No intermediate LLM call - agent IS the LLM!
            code = pde_code

            # Fit parameters and evaluate using the code
            from scipy import optimize

            # Fast objective: match dg/dt across all time steps (vectorized)
            g_series = problem.g_observed
            dt_meta = float(problem.metadata.get('dt', 0.01))

            last_obj_error: Optional[str] = None
            def objective(params):
                nonlocal last_obj_error
                g_pred, dgdt_pred, success, error = self.llmsr_solver.evaluate_pde_dgdt(
                    code, g_series, problem.S, params
                )
                if not success or dgdt_pred is None:
                    last_obj_error = str(error)
                    if getattr(self, 'verbose_debug', False):
                        print(f"[Objective] dgdt eval failed: {last_obj_error}")
                    return 1e10
                dgdt_obs = (g_series[:, :, 1:] - g_series[:, :, :-1]) / dt_meta
                return float(np.mean((dgdt_pred - dgdt_obs) ** 2))

            # Initial guess: midpoint of bounds
            x0 = [(b[0] + b[1]) / 2 for b in param_bounds]
            result = optimize.minimize(objective, x0, method='L-BFGS-B', bounds=param_bounds, )
            fitted_params_list = result.x
            loss = result.fun
            steps_roll = min(g_series.shape[2], self.eval_max_steps)
            rollout_pred, rsuccess, rerror = self.llmsr_solver.evaluate_pde(
                code, problem.g_init, problem.S, fitted_params_list, steps_roll
            )

            if rsuccess and rollout_pred is not None:
                g_obs_roll = g_series[:, :, :steps_roll]
                predicted_for_viz = rollout_pred
            else:
                if getattr(self, 'verbose_debug', False):
                    print(f"[Rollout] evaluate_pde failed: {rerror}")
                # Fallback to one-step (teacher-forced) sequence
                g_obs_roll = g_series
                # Make sure we have g_pred; recompute quickly if needed
                if 'g_pred' not in locals() or g_pred is None:
                    g_pred, _, _, _ = self.llmsr_solver.evaluate_pde_dgdt(
                        code, g_series, problem.S, fitted_params_list
                    )
                predicted_for_viz = g_pred if g_pred is not None else g_series.copy()

            # R² on rollout/fallback
            ss_res = np.sum((g_obs_roll - predicted_for_viz) ** 2)
            ss_tot = np.sum((g_obs_roll - g_obs_roll.mean()) ** 2)
            r2 = float(1 - ss_res / (ss_tot + 1e-10)) if ss_tot > 0 else 0.0

            # Mass error final frame
            final_mass_pred = float(predicted_for_viz[:, :, -1].sum())
            final_mass_obs = float(g_obs_roll[:, :, -1].sum())
            mass_error = float(abs(final_mass_pred - final_mass_obs) / (final_mass_obs + 1e-8) * 100)

            fitted_params = {f'p{i}': val for i, val in enumerate(fitted_params_list)}

            # NMSE on rollout/fallback
            nmse = float(self.solver.compute_spatiotemporal_loss(predicted_for_viz, g_obs_roll, 'nmse'))

            # Combined numerical score from R^2 (rollout) and MSE (dg/dt objective)
            # - r2_score in [0,1]
            # - mse_score = 1 / (1 + normalized_mse) where normalized by dg/dt energy
            r2_score = max(0.0, min(1.0, float(r2)))
            # Recompute observed dg/dt to scale MSE robustly
            dgdt_obs_full = (g_series[:, :, 1:] - g_series[:, :, :-1]) / dt_meta
            dgdt_scale = float(np.mean(dgdt_obs_full ** 2)) + 1e-8
            mse_norm = float(loss) / dgdt_scale
            mse_score = 1.0 / (1.0 + mse_norm)
            # Blend weights (can be tuned)
            w_r2 = 0.5
            blended = w_r2 * r2_score + (1.0 - w_r2) * mse_score
            # Apply mass penalty and scale to [0,10]
            mass_penalty = (1 - min(mass_error / 100.0, 0.5))
            numerical_score = max(0.0, min(10.0, 10.0 * blended * mass_penalty))

            # Always generate visualization for critic
            self.eval_counter += 1
            viz_path = self.viz_dir / f"iter_{self.iteration:06d}_eval_{self.eval_counter:04d}.png"

            # Search for the first PDE FORM comment in the code (before \n)

            import re
            code_summary = re.search(r'# PDE FORM: (.*)', code).group(1).split('\n')[0].strip()
            if code_summary is None:
                code_summary = code

            self.visualizer.create_critique_visualization(
                g_obs_roll, predicted_for_viz,
                code_summary, {'mse': loss, 'r2': r2, 'nmse': nmse, 'mass_error': mass_error},
                save_path=str(viz_path)
            )

            # Do not run visual critic here; discover() will use critic_agent with the saved viz

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar('metrics/score', numerical_score, self.iteration)
                self.writer.add_scalar('metrics/r2', r2, self.iteration)
                self.writer.add_scalar('metrics/mse', loss, self.iteration)
                self.writer.add_scalar('metrics/mass_error', mass_error, self.iteration)

            # Buffer add deferred to after visual critic

            # Optional debug print for each evaluation
            if getattr(self, 'verbose_debug', False):
                print(f"[Eval] Iter {self.iteration} code: {code_summary} | Num={numerical_score:.2f}")

            # # Update best
            if numerical_score > self.numerical_best_score:
                self.numerical_best_score = numerical_score
                self.numerical_best_equation = code_summary  # Store code summary

                if self.writer:
                    self.writer.add_scalar('num_best/score', numerical_score, self.iteration)
                    self.writer.add_scalar('num_best/r2', r2, self.iteration)
                    self.writer.add_scalar('num_best/mse', loss, self.iteration)
                    self.writer.add_scalar('num_best/nmse', nmse, self.iteration)
                    self.writer.add_scalar('num_best/mass_error', mass_error, self.iteration)

                print(f"\n🎯 Iter {self.iteration}: NEW BEST (numerical)! Num={numerical_score:.4f} R²={r2:.4f}")
                print(f"   Code: {code_summary}")
                print(f"   Ground Truth: {problem.gt_equation}")
                print(f"   Params: {fitted_params}")


            result_dict = {
                'success': True,
                'score': float(numerical_score),  # numerical score (backward-compat)
                'numerical_score': float(numerical_score),
                'r2': float(r2),
                'mse': float(loss),
                'nmse': float(nmse),
                'mass_error': float(mass_error),
                'fitted_params': fitted_params,
                'num_params': num_params,
                'visual_score': None,
                'combined_score': None,
                'visualization_path': str(viz_path),
                'visual_summary': None,
                'code_preview': code[:2500] if len(code) > 2500 else code,
                'PDE': code_summary,
                'last_objective_error': last_obj_error,
                'rollout_error': rerror if not rsuccess else None,
                'critic': None,
                'message': (
                    f"✓ Numerical={numerical_score:.2f} [AGENT-CODE]. R²={r2:.4f}, MSE={loss:.6f}, MassErr={mass_error:.2f}%.\n"
                    "Next: use critic_agent with viz_path for visual analysis."
                )
            }

            return json.dumps(result_dict)

        except Exception as e:
            import traceback
            error_msg = str(e)
            # Provide more helpful error messages
            if "not yet supported" in error_msg.lower():
                error_msg = f"PDE form not supported. {error_msg}\n\nPlease use only supported forms:\n{SUPPORTED_FORMS_GUIDE}"

            error_result = {
                'success': False,
                'error': error_msg,
                'score': 0.0,
                'message': f"✗ Evaluation failed: {error_msg}"
            }
            if getattr(self, 'verbose_debug', False):
                print(f"[Eval Error] PDE evaluation failed")
                print(f"  Error: {error_msg}")
                traceback.print_exc()
            return json.dumps(error_result)

    async def discover(self, problem, verbose: bool = True):
        """Main discovery loop with AutoGen v0.4"""

        self.current_problem = problem
        H, W, T = problem.g_observed.shape
        # Enable class-wide debug printing for tool if verbose
        self.verbose_debug = bool(verbose)

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

        # Ensure LLMSR solver uses dataset's dx, dy, dt (avoid mismatch)
        try:
            self.llmsr_solver.dx = float(data_summary['dx'])
            self.llmsr_solver.dy = float(data_summary['dy'])
            self.llmsr_solver.dt = float(data_summary['dt'])
            if verbose:
                print(f"[Debug] LLMSR solver grid set: dx={self.llmsr_solver.dx}, dy={self.llmsr_solver.dy}, dt={self.llmsr_solver.dt}")
        except Exception as _e:
            if verbose:
                print(f"[Warn] Failed to set LLMSR solver grid from metadata: {_e}")
            print(f"Dataset: {data_summary['shape']}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Ground Truth: {problem.gt_equation}")
            print(f"Mass change: {data_summary['mass_change_pct']:.2f}%")
            print("="*70)

        # Build PDE Generator agent with tool
        generator_agent = AssistantAgent(
            name="PDE_Generator",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=2),
            tools=[],
            system_message=f"""You are an expert at discovering spatiotemporal PDEs from observational data.

{SUPPORTED_FORMS_GUIDE}

YOUR TASK:
You generate PDE hypotheses and get them evaluated. A Visual Critic will analyze the results.

For each round, propose {self.samples_per_prompt} DIVERSE PDE hypotheses as complete Python code.


CRITICAL RULES:
1. Generate complete pde_update() function code using scipy.ndimage for operators
2. Focus on moderately optimizing the best candidate, instead of involving too many terms. The generated PDE MUST have less than 6 term 
3. Try DIVERSE hypotheses each round (diffusion, reaction, advection, nonlinear, etc.)
4. Wait for Visual Critic feedback before proposing next batch
6. Focus on: High R² (>0.9), Low mass error (<5%), High visual score
7. You MUST comment the intermediate variables' shape to avoid errors.
8. At the beginning, you MUST comment the PDE FORM in the code.

WORKFLOW PER ROUND:
1. Propose {self.samples_per_prompt} DIFFERENT PDE code implementations
2. Wait for Visual Critic analysis
3. Learn from critic feedback and propose better hypotheses next round
""",
        )

        # Build Visual Critic agent (vision-enabled)
        critic_agent = AssistantAgent(
            name="Visual_Critic",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=2),
            tools=[],
            system_message="""You are a Visual Critic specialized in analyzing PDE simulation results.

YOUR TASK:
Analyze the visualization plots and evaluation metrics for PDE candidates.
Provide constructive feedback to the PDE_Generator.

ANALYSIS FOCUS:
1. Spatial pattern accuracy (do predicted patterns match observations?)
2. Temporal evolution (does the dynamics match over time?)
3. Mass conservation (is total mass preserved?)
4. Boundary behavior (proper handling of domain edges?)
5. Physical plausibility (makes sense for the system?)

FEEDBACK FORMAT:
For each evaluated PDE, provide:
- Key observations about what works/doesn't work
- Specific suggestions for improvement (e.g., "increase diffusion", "add reaction term")
- Ranking of current batch (which was best and why)
- Direction for next iteration

Be concise but specific. Focus on actionable feedback.""",
        )
        # Expose critic_agent to tools if needed
        self.critic_agent = critic_agent

        # Discovery loop
        for iteration in range(1, self.max_iterations + 1):
            self.iteration = iteration
            iter_start = time.time()

            # Get top-3 experience context (include visual feedback)
            experience_context = self.buffer.format_for_prompt(k=3, include_visual=False)

            # Create task for this round - directed at Generator
            if experience_context:
                task_text = f"""
DATA SUMMARY:
- Grid: {data_summary['shape']} (H×W×T), mass change: {data_summary['mass_change_pct']:.2f}%

PREVIOUS TOP RESULTS (with Visual Critic):
{experience_context}

PDE_Generator: Generate {self.samples_per_prompt} new DIVERSE PDE code implementations and evaluate each.
Learn from previous results and visual critic feedback. Generate COMPLETE pde_update() functions using scipy.ndimage operators.
"""
            else:
                task_text = f"""
DATA SUMMARY:
- Grid: {data_summary['shape']} (H×W×T), mass change: {data_summary['mass_change_pct']:.2f}%

First round - PDE_Generator, please explore diverse PDE hypotheses:
- Try {self.samples_per_prompt} DIFFERENT PDE implementations as complete Python code
- Vary complexity (1-5 parameters)
- Use scipy.ndimage for Laplacian, gradients, etc.
- Examples: pure diffusion, diffusion+reaction, diffusion+chemotaxis, nonlinear terms
- DO NOT assume any specific form - let the data guide you!
"""

            # Phase 1: Generator proposes and evaluates PDEs
            try:
                cancellation_token = CancellationToken()
                prev_buf_size = len(self.buffer)

                # Generator generates and evaluates
                generator_result = await generator_agent.on_messages(
                    [TextMessage(content=task_text, source="user")],
                    cancellation_token=cancellation_token
                )

                if verbose:
                    try:
                        content = getattr(generator_result.chat_message, 'content', '')
                        # print(f"\n[Iter {iteration}] Generator response preview: {str(content)}")
                    except Exception as _e:
                        print(f"[Debug] Failed to print response: {_e}")

                # Parse generator response and extract pde_update codes
                try:
                    response_content = getattr(generator_result.chat_message, 'content', '')
                    # Extract all code blocks between ```python and ```
                    import re
                    code_pattern = r'```python\s*(.*?)```'
                    code_blocks = re.findall(code_pattern, response_content, re.DOTALL)
                    eval_results = []
                    
                    if verbose:
                        print(f"\n[Iter {iteration}] Extracted {len(code_blocks)} code blocks from generator")
                    # Evaluate each extracted pde_update code
                    for idx, pde_code in enumerate(code_blocks):                        
                        # Try to infer parameter count by looking for params[0], params[1], etc.
                        param_usage = re.findall(r'params\[(\d+)\]', pde_code)
                        if param_usage:
                            num_params = max([int(i) for i in param_usage]) + 1
                        
                        if verbose:
                            print(f"\n[Iter {iteration}] Evaluating PDE #{idx+1}/{len(code_blocks)} (num_params={num_params})")
                        
                        # Call evaluate_pde_tool
                        eval_result = json.loads(self.evaluate_pde(pde_code, num_params=num_params))
                        eval_results.append(eval_result)
                        if verbose:
                            try:
                                score = eval_result.get('score', 0)
                                print(f"[Iter {iteration}] PDE #{idx+1} {eval_result.get('PDE', '')} evaluated: Score={score:.4f}")
                            except:
                                print(f"[Iter {iteration}] PDE #{idx+1} {eval_result.get('PDE', '')} evaluated")
                
                except Exception as parse_e:
                    if verbose:
                        print(f"[Iter {iteration}] Failed to parse generator response: {parse_e}")
                        import traceback
                        traceback.print_exc()
                
                # Phase 2: Visual Critic analyzes each evaluated PDE (per-image, multimodal)
                # For each eval, call critic_agent with image and metrics, then update buffer
            
                for idx, eval_result in enumerate(eval_results, 1):
                    try:
                        viz_path = eval_result.get('visualization_path')
                        pde_equation = eval_result.get('PDE', '')
                        code_preview = eval_result.get('code_preview', '')
                        metrics = eval_result.get('metrics', {})
                        # Build multimodal message
                        img = PILImage.open(viz_path) if viz_path and Path(viz_path).exists() else None
                        if img is None:
                            if verbose:
                                print(f"[Critic] Missing viz image for eval #{idx}, skipping visual analysis")
                            continue
                        mm = MultiModalMessage(
                            content=[
                                (
                                    "Analyze the PDE simulation visualization (pred vs obs). Return JSON: "
                                    "{score:0-10, spatial_analysis, temporal_analysis, boundary_analysis, "
                                    "conservation_assessment, suggestions}.\n"
                                    f"Code: {code_preview}\n"
                                    f"Metrics: {metrics}"
                                ),
                                Image(img),
                            ],
                            source="user",
                        )
                        result = await critic_agent.run(task=mm)
                        content = result.messages[-1].content
                        try:
                            cj = json.loads(content)
                        except Exception:
                            cj = {'freeform': content}

                        visual_score = cj.get('score') if isinstance(cj, dict) else None
                        spatial = cj.get('spatial_analysis', '') if isinstance(cj, dict) else ''
                        temporal = cj.get('temporal_analysis', '') if isinstance(cj, dict) else ''
                        boundary = cj.get('boundary_analysis', '') if isinstance(cj, dict) else ''
                        conservation = cj.get('conservation_assessment', '') if isinstance(cj, dict) else ''
                        suggestions = cj.get('suggestions', []) if isinstance(cj, dict) else []
                        if isinstance(suggestions, str):
                            suggestions = [suggestions]

                        visual_summary = (
                            f"Visual Critic (score {visual_score}/10)\n"
                            + (f"- Spatial: {spatial[:250]}\n" if spatial else "")
                            + (f"- Temporal: {temporal[:250]}\n" if temporal else "")
                            + (f"- Boundary: {boundary[:250]}\n" if boundary else "")
                            + (f"- Conservation: {conservation[:250]}\n" if conservation else "")
                            + (f"- Key Issue: {suggestions if suggestions else 'None'}\n")
                        )

                        # Combine scores
                        num_score = eval_result.get('numerical_score', 0.0)
                        w = self.visual_weight
                        combined = float((1.0 - w) * float(num_score) + w * float(visual_score)) if isinstance(visual_score, (int, float)) else float(num_score)

                        # Add to buffer
                        self.buffer.add(
                            equation=eval_result.get('code_preview', ''),
                            score=float(num_score),
                            metrics={
                                'mse': eval_result.get('mse'), 'r2': eval_result.get('r2'), 'nmse': eval_result.get('nmse'), 'mass_error': eval_result.get('mass_error'),
                                'visual_score': visual_score, 'combined_score': combined
                            },
                            visual_analysis=visual_summary,
                            reasoning="Agent-generated code",
                            suggestions='; '.join(suggestions),
                            parameters=metrics.get('fitted_params', {}),
                            visualization_path=viz_path,
                            spatial_assessment=spatial,
                            temporal_assessment=temporal,
                            visual_score=visual_score,
                            combined_score=combined,
                        )

                        # Update best with combined
                        # if combined > self.best_score:
                        #     self.best_score = combined
                        #     self.best_equation = pde_equation
                        #     self.plateau_counter = 0
                        #     if self.writer:
                        #         self.writer.add_scalar('best/combined_score', combined, self.iteration)
                        #     if verbose:
                        #         print(f"[Iter {iteration}] NEW BEST combined={combined:.2f} (num={num_score:.2f}, vis={visual_score})")
                    except Exception as ce:
                        if verbose:
                            print(f"[Critic] Failed to analyze eval #{idx}: {ce}")

            except Exception as e:
                if verbose:
                    print(f"Iteration {iteration} failed: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            # Log iteration time
            iter_time = time.time() - iter_start
            if self.writer:
                self.writer.add_scalar('performance/iteration_time', iter_time, iteration)
                self.writer.add_scalar('performance/buffer_size', len(self.buffer), iteration)
                self.writer.add_scalar('performance/plateau_counter', self.plateau_counter, iteration)

            # Progress reporting
            if verbose and iteration % 1 == 0:
                elapsed = time.time() - start_time
                print(f"\n{'='*70}")
                print(f"Progress: Iteration {iteration}/{self.max_iterations}")
                print(f"Numerical Best Score: {self.numerical_best_score:.4f} | Plateau: {self.plateau_counter}/{self.plateau_patience}")
                print(f"Buffer Size: {len(self.buffer)} PDEs | Elapsed: {elapsed:.1f}s")
                if self.numerical_best_equation:
                    print(f"Numerical Best Equation: {self.numerical_best_equation}")
                    print(f"Ground Truth: {problem.gt_equation}")
                print(f"{'='*70}")

            # Convergence check (combined score scaled to 0-10)
            if self.numerical_best_score >= self.convergence_threshold * 10:
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
            'success': self.numerical_best_equation is not None,
            'numerical_best_equation': self.numerical_best_equation,
            'numerical_best_score': float(self.numerical_best_score),
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
            print(f"Numerical Best Equation: {self.numerical_best_equation}")
            print(f"Numerical Best Score: {self.numerical_best_score:.4f}")
            print(f"Total Iterations: {iteration}")
            print(f"Total Time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
            print(f"Results saved to: {results_path}")
            if self.writer:
                print(f"\nTensorBoard: tensorboard --logdir {self.output_dir / 'tensorboard'} --port 6006")
            print("="*70)

        return results


async def main_async():
    parser = argparse.ArgumentParser(description="PDE Discovery - AutoGen v0.4")
    parser.add_argument('--dataset', type=str, default='logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5', help='Path to dataset (HDF5) or folder with npy pair')
    parser.add_argument('--ca_path', type=str, default=None, help='Path to npy: chemoattractant (S); expects shape (T,H,W)')
    parser.add_argument('--cell_path', type=str, default=None, help='Path to npy: cell density video (g); expects shape (T,H,W)')
    parser.add_argument('--from_npy', action='store_true', help='Load from npy files instead of HDF5')
    parser.add_argument('--fit_fixed', action='store_true', help='Bypass LLM; fit fixed PDE ∂g/∂t = D·Δg - χ·∇·(g·∇ln(S)) + r·g·(1-g/K)')
    parser.add_argument('--dt', type=float, default=1.0, help='Time step for npy data (default: 1.0)')
    parser.add_argument('--dx', type=float, default=1.0, help='Pixel size x (default: 1.0)')
    parser.add_argument('--dy', type=float, default=1.0, help='Pixel size y (default: 1.0)')
    parser.add_argument('--n_restarts', type=int, default=8, help='Number of random restarts for fixed PDE fitting')
    parser.add_argument('--max_iter_fit', type=int, default=200, help='Max iterations for optimizer in fixed PDE fitting')
    parser.add_argument('--api_base', type=str, default='http://localhost:10005/v1', help='vLLM API URL')
    parser.add_argument('--api_model', type=str, default='/mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct', help='Model path')
    parser.add_argument('--max_iterations', type=int, default=8000, help='Max iterations')
    parser.add_argument('--samples_per_prompt', type=int, default=4, help='Samples per prompt')
    parser.add_argument('--output_dir', type=str, default='./logs/pde_discovery_autogen_v04', help='Output directory')
    parser.add_argument('--critic_model', type=str, default='/mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct', help='Vision model for Visual Critic (defaults to api_model)')
    args = parser.parse_args()

    # Helper: loader for npy pair
    def _load_npy_pair(ca_path: Optional[str], cell_path: Optional[str]) -> ChemotaxisProblem:
        # If folder is given in --dataset, assume default filenames
        if (ca_path is None or cell_path is None) and args.dataset and Path(args.dataset).is_dir():
            folder = Path(args.dataset)
            ca_default = folder / 'ca_video_continuous.npy'
            cell_default = folder / 'cell_video_continuous.npy'
            if ca_path is None:
                ca_path = str(ca_default)
            if cell_path is None:
                cell_path = str(cell_default)

        if ca_path is None or cell_path is None:
            raise ValueError('Both --ca_path and --cell_path must be provided (or set --dataset to folder containing the default npy filenames).')

        print(f"Loading npy pair: S={ca_path}, g={cell_path}")
        S_np = np.load(ca_path)
        g_np = np.load(cell_path)
        if S_np.ndim != 3 or g_np.ndim != 3:
            raise ValueError(f"Expected (T,H,W) arrays; got S{S_np.shape}, g{g_np.shape}")

        # Convert (T,H,W) -> (H,W,T)
        S = np.transpose(S_np, (1, 2, 0)).astype(np.float32)
        g_series = np.transpose(g_np, (1, 2, 0)).astype(np.float32)

        # Standardize by std only (per user request)
        S_std = float(S.std())
        g_std = float(g_series.std())
        S = S / max(S_std, 1e-8)
        g_series = g_series / max(g_std, 1e-8)

        # Numerical safety for ln(S)
        S = np.maximum(S, 1e-6)

        metadata = {
            'dx': float(args.dx), 'dy': float(args.dy), 'dt': float(args.dt),
            'source': 'npy_pair',
            'reference_pde': '∂g/∂t = D·Δg - χ·∇·(g·∇ln(S)) + r·g·(1-g/K)'
        }
        problem = ChemotaxisProblem(
            g_init=g_series[:, :, 0].copy(),
            S=S,
            g_observed=g_series,
            metadata=metadata,
            gt_equation=metadata['reference_pde']
        )
        return problem

    # Load dataset
    if args.from_npy or args.ca_path or args.cell_path or (args.dataset and Path(args.dataset).is_dir()):
        problem = _load_npy_pair(args.ca_path, args.cell_path)
        print(f"✓ Loaded (npy): {problem.g_observed.shape}")
    else:
        print(f"Loading dataset: {args.dataset}")
        dm = ChemotaxisDataModule(data_source="hdf5", data_path=args.dataset)
        problems = dm.load()
        problem = list(problems.values())[0]
        print(f"✓ Loaded (hdf5): {problem.g_observed.shape}")
    print(f"  Ground Truth / Reference: {problem.gt_equation}")

    # If requested, run a fixed-PDE fitting using the provided reference form
    if args.fit_fixed:
        # Fit parameters on dg/dt objective with rollout validation
        print("Running fixed-PDE fitting (D, chi, r, K)...")

        H, W, T = problem.g_observed.shape
        dx = float(problem.metadata.get('dx', args.dx))
        dy = float(problem.metadata.get('dy', args.dy))
        dt = float(problem.metadata.get('dt', args.dt))

        g_series = problem.g_observed.astype(np.float32)
        S_series = problem.S.astype(np.float32)

        # Precompute observed dg/dt
        dgdt_obs = (g_series[:, :, 1:] - g_series[:, :, :-1]) / dt

        # Utility: vectorized spatial ops on (H,W,T-1)
        def grad_x(a):
            import scipy.ndimage as nd
            return nd.sobel(a, axis=1) / (2.0 * dx)

        def grad_y(a):
            import scipy.ndimage as nd
            return nd.sobel(a, axis=0) / (2.0 * dy)

        def laplacian_xy(a):
            # Second derivatives only along x and y, vectorized over time dim
            d2x = np.gradient(np.gradient(a, dx, axis=1), dx, axis=1)
            d2y = np.gradient(np.gradient(a, dy, axis=0), dy, axis=0)
            return d2x + d2y

        # Predicted dg/dt for all t using current state g_t and S_t (T-1 slices)
        def predict_dgdt(params):
            D, chi, r, K = params
            g_t = g_series[:, :, :-1]
            S_t = S_series[:, :, :-1] if S_series.ndim == 3 else S_series
            # Ensure broadcast if S is 2D
            if S_series.ndim == 2:
                S_t = np.repeat(S_series[:, :, None], g_t.shape[2], axis=2)
            S_t = np.maximum(S_t, 1e-6)

            lap_g = laplacian_xy(g_t)
            lnS = np.log(S_t)
            grad_lnS_x = grad_x(lnS)
            grad_lnS_y = grad_y(lnS)
            flux_x = g_t * grad_lnS_x
            flux_y = g_t * grad_lnS_y
            div_flux = grad_x(flux_x) + grad_y(flux_y)
            reaction = r * g_t * (1.0 - g_t / (K + 1e-8))
            return D * lap_g - chi * div_flux + reaction

        # Objective: MSE between predicted and observed dg/dt
        def objective(params):
            D, chi, r, K = params
            # Enforce positivity for D, chi, K; r allowed >= 0
            if D < 0 or chi < 0 or K <= 0 or r < 0:
                return 1e12
            pred = predict_dgdt(params)
            return float(np.mean((pred - dgdt_obs) ** 2))

        # Parameter bounds (coarse): D in [0, 5], chi in [0, 50], r in [0, 2], K in [0.25*max, 4*max]
        g_max = float(np.max(g_series))
        bounds = [(1e-6, 5.0), (0.0, 50.0), (0.0, 2.0), (max(1e-3, 0.25 * g_max), max(1.0, 4.0 * g_max))]

        # Multi-start optimization to estimate parameter ranges
        from scipy.optimize import minimize
        rng = np.random.default_rng(42)
        best_res = None
        trials = []
        for i in range(int(args.n_restarts)):
            x0 = np.array([
                rng.uniform(*bounds[0]),
                rng.uniform(*bounds[1]),
                rng.uniform(*bounds[2]),
                rng.uniform(*bounds[3]),
            ], dtype=np.float64)
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={"maxiter": int(args.max_iter_fit)})
            trials.append(res.x.tolist() + [res.fun])
            if best_res is None or res.fun < best_res.fun:
                best_res = res
            print(f"  Restart {i+1}/{args.n_restarts}: loss={res.fun:.6e}, params=[D={res.x[0]:.4g}, chi={res.x[1]:.4g}, r={res.x[2]:.4g}, K={res.x[3]:.4g}]")

        trials = np.array(trials)
        D_list, chi_list, r_list, K_list, loss_list = trials.T
        summary = {
            'D': {'min': float(D_list.min()), 'max': float(D_list.max()), 'mean': float(D_list.mean()), 'median': float(np.median(D_list))},
            'chi': {'min': float(chi_list.min()), 'max': float(chi_list.max()), 'mean': float(chi_list.mean()), 'median': float(np.median(chi_list))},
            'r': {'min': float(r_list.min()), 'max': float(r_list.max()), 'mean': float(r_list.mean()), 'median': float(np.median(r_list))},
            'K': {'min': float(K_list.min()), 'max': float(K_list.max()), 'mean': float(K_list.mean()), 'median': float(np.median(K_list))},
            'best_loss': float(best_res.fun),
        }

        print("\nEstimated parameter ranges (across restarts):")
        for k, v in summary.items():
            if k == 'best_loss':
                continue
            print(f"  {k}: min={v['min']:.4g}, max={v['max']:.4g}, mean={v['mean']:.4g}, median={v['median']:.4g}")
        print(f"  best_loss (MSE_dgdt): {summary['best_loss']:.6e}")

        # Rollout using best params
        best_params = best_res.x

        def one_step(g2d, S2d, params):
            D, chi, r, K = params
            # 2D ops
            import scipy.ndimage as nd
            lap = nd.laplace(g2d) / (dx * dx)
            lnS = np.log(np.maximum(S2d, 1e-6))
            gx = nd.sobel(lnS, axis=1) / (2.0 * dx)
            gy = nd.sobel(lnS, axis=0) / (2.0 * dy)
            div = (nd.sobel(g2d * gx, axis=1) / (2.0 * dx)) + (nd.sobel(g2d * gy, axis=0) / (2.0 * dy))
            react = r * g2d * (1.0 - g2d / (K + 1e-8))
            dgdt = D * lap - chi * div + react
            return np.maximum(g2d + dt * dgdt, 0.0)

        Tn = g_series.shape[2]
        g_roll = np.zeros_like(g_series)
        g_roll[:, :, 0] = g_series[:, :, 0]
        for t in range(1, Tn):
            S_t = S_series[:, :, t] if S_series.ndim == 3 else S_series
            g_roll[:, :, t] = one_step(g_roll[:, :, t-1], S_t, best_params)

        # Save 9-panel visualization
        out_dir = Path(args.output_dir) / 'npy_fit'
        out_dir.mkdir(parents=True, exist_ok=True)
        viz_path = out_dir / 'rollout_6x3.png'
        PDEVisualizer().create_rollout_grid_with_stats(
            observed=g_series, predicted=g_roll, save_path=str(viz_path)
        )
        print(f"Saved rollout and stats visualization to: {viz_path}")

        # Save parameter summary
        with open(out_dir / 'fit_summary.json', 'w') as f:
            json.dump({
                'best_params': {'D': float(best_params[0]), 'chi': float(best_params[1]), 'r': float(best_params[2]), 'K': float(best_params[3])},
                'ranges': summary,
            }, f, indent=2)
        print(f"Saved parameter summary to: {out_dir / 'fit_summary.json'}")

        return { 'best_params': best_params.tolist(), 'ranges': summary }

    # Run discovery (LLM-driven) if not fixed fitting
    if args.fit_fixed:
        # In fixed-fit mode, we already returned above after completion
        return

    system = PDEDiscoveryAutogenV04(
        api_base=args.api_base,
        api_model=args.api_model,
        critic_model=args.critic_model,
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
    print(f"Discovered:   {results['numerical_best_equation']}")
    print(f"\nGT Parameters: {results['gt_parameters']}")
    # Try to show fitted params for discovered best equation (by combined score)
    fitted_params = None
    for exp in system.buffer.experiences:
        if exp.equation == results['numerical_best_equation']:
            fitted_params = exp.parameters
            break
    if fitted_params is None and system.buffer.get_best():
        fitted_params = system.buffer.get_best().parameters
    if fitted_params is not None:
        print(f"Fitted Params: {fitted_params}")
    print("="*70)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
