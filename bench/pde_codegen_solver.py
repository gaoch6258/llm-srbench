"""
Code Generation PDE Solver - Tier 2 Fallback

This module handles exotic PDEs that cannot be parsed symbolically.
It uses LLM to generate executable code for iterative PDE updates,
similar to the LLMSR approach.

Safety:
- Subprocess isolation
- Timeout enforcement
- Namespace restrictions
- Output validation
"""

import numpy as np
import multiprocessing
import ast
import re
import traceback
from typing import Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import time
import sys
from io import StringIO


@dataclass
class GeneratedSolver:
    """Container for generated PDE solver code"""
    source_code: str
    function_name: str
    description: str
    parameters: list[str]
    success: bool = False
    error: Optional[str] = None


class CodeExecutionSandbox:
    """Secure sandbox for executing generated PDE solver code"""

    def __init__(self, timeout_seconds: int = 30):
        self.timeout = timeout_seconds

    @staticmethod
    def _execute_in_subprocess(code: str, function_name: str,
                               g_init: np.ndarray, S: np.ndarray,
                               params: np.ndarray, dt: float, dx: float, dy: float,
                               num_steps: int, result_queue: multiprocessing.Queue):
        """Execute generated code in isolated subprocess"""
        try:
            # Capture stdout/stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()

            # Create restricted namespace
            namespace = {
                'np': np,
                'numpy': np,
                '__builtins__': {
                    'range': range,
                    'len': len,
                    'min': min,
                    'max': max,
                    'abs': abs,
                    'sum': sum,
                    'enumerate': enumerate,
                    'zip': zip,
                    'float': float,
                    'int': int,
                    'bool': bool,
                    'True': True,
                    'False': False,
                    'None': None,
                },
            }

            # Execute code to define function
            exec(code, namespace)

            # Get the function
            if function_name not in namespace:
                result_queue.put((None, False, f"Function {function_name} not found in code"))
                return

            pde_update_func = namespace[function_name]

            # Run time integration
            H, W = g_init.shape
            g_history = np.zeros((H, W, num_steps))
            g_history[:, :, 0] = g_init.copy()
            g_current = g_init.copy()

            for t in range(1, num_steps):
                # Get current S field
                if S.ndim == 3:
                    S_current = S[:, :, min(t, S.shape[2]-1)]
                else:
                    S_current = S

                # Call generated update function
                g_next = pde_update_func(
                    g=g_current,
                    S=S_current,
                    dx=dx,
                    dy=dy,
                    dt=dt,
                    params=params
                )

                # Validate output
                if not isinstance(g_next, np.ndarray):
                    result_queue.put((None, False, f"Function must return np.ndarray, got {type(g_next)}"))
                    return

                if g_next.shape != g_current.shape:
                    result_queue.put((None, False, f"Shape mismatch: expected {g_current.shape}, got {g_next.shape}"))
                    return

                # Check for NaN/Inf
                if np.any(np.isnan(g_next)) or np.any(np.isinf(g_next)):
                    result_queue.put((None, False, f"NaN or Inf detected at timestep {t}"))
                    return

                # Physical constraints
                g_next = np.maximum(g_next, 0)  # Non-negativity

                g_history[:, :, t] = g_next
                g_current = g_next

            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            result_queue.put((g_history, True, None))

        except Exception as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result_queue.put((None, False, error_msg))

    def execute(self, code: str, function_name: str,
                g_init: np.ndarray, S: np.ndarray,
                params: Dict[str, float], dx: float, dy: float, dt: float,
                num_steps: int) -> Tuple[Optional[np.ndarray], bool, Optional[str]]:
        """
        Execute generated code in sandbox

        Args:
            code: Python source code
            function_name: Name of the update function
            g_init: Initial condition
            S: Signal field
            params: Parameter values
            dx, dy, dt: Discretization parameters
            num_steps: Number of time steps

        Returns:
            (result, success, error_message)
        """
        # Convert params dict to array
        param_array = np.array(list(params.values()), dtype=float)

        # Create result queue
        result_queue = multiprocessing.Queue()

        # Start subprocess
        process = multiprocessing.Process(
            target=self._execute_in_subprocess,
            args=(code, function_name, g_init, S, param_array, dt, dx, dy, num_steps, result_queue)
        )

        process.start()
        process.join(timeout=self.timeout)

        # Check if process is still alive (timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            return None, False, f"Execution timeout ({self.timeout}s exceeded)"

        # Get result from queue
        if not result_queue.empty():
            result, success, error = result_queue.get()
            return result, success, error
        else:
            return None, False, "Process terminated unexpectedly"


class PDECodeGenerator:
    """Generate PDE solver code using LLM or templates"""

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Args:
            llm_client: Optional LLM client for code generation
                       If None, uses template-based generation
        """
        self.llm_client = llm_client

    def extract_function_from_response(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract function body from LLM response

        Args:
            response: LLM response text

        Returns:
            (function_body, function_name) or (None, None) if extraction fails
        """
        # Look for def pde_update or similar
        pattern = r'def\s+(\w+)\s*\([^)]*\)\s*(?:->.*?)?:'

        match = re.search(pattern, response)
        if not match:
            return None, None

        function_name = match.group(1)
        start_idx = match.start()

        # Extract the full function using AST parsing
        try:
            # Try to parse from this point
            code_from_def = response[start_idx:]

            # Find the end of the function by parsing
            tree = ast.parse(code_from_def)
            if tree.body and isinstance(tree.body[0], ast.FunctionDef):
                func_node = tree.body[0]
                # Get source lines
                lines = code_from_def.split('\n')
                # The function ends at the last line of its body
                end_line = max(node.end_lineno for node in ast.walk(func_node) if hasattr(node, 'end_lineno'))
                function_body = '\n'.join(lines[:end_line])
                return function_body, function_name

        except SyntaxError:
            # Fallback: extract until dedent
            lines = response[start_idx:].split('\n')
            func_lines = [lines[0]]  # def line

            for line in lines[1:]:
                if line and not line[0].isspace() and line.strip():
                    # Dedented non-empty line, function ended
                    break
                func_lines.append(line)

            return '\n'.join(func_lines), function_name

        return None, None

    def generate_from_template(self, pde_description: str, param_names: list[str]) -> GeneratedSolver:
        """
        Generate solver code from template (fallback when no LLM)

        Args:
            pde_description: Natural language PDE description
            param_names: List of parameter names

        Returns:
            GeneratedSolver object
        """
        # Create a generic template
        params_str = ', '.join(f"params[{i}]  # {name}" for i, name in enumerate(param_names))

        template = f'''
def pde_update(g: np.ndarray, S: np.ndarray, dx: float, dy: float, dt: float, params: np.ndarray) -> np.ndarray:
    """
    Generated PDE update function

    PDE: {pde_description}

    Args:
        g: Current density field (H, W)
        S: Signal field (H, W)
        dx, dy: Spatial resolution
        dt: Time step
        params: [{params_str}]

    Returns:
        Updated g field (H, W)
    """
    # Extract parameters
    {chr(10).join(f"    {name} = params[{i}]" for i, name in enumerate(param_names))}

    # Compute spatial operators (5-point stencil Laplacian, central difference gradient)
    # Laplacian of g
    laplacian_g = np.zeros_like(g)
    laplacian_g[1:-1, 1:-1] = (
        (g[1:-1, 2:] - 2*g[1:-1, 1:-1] + g[1:-1, :-2]) / dx**2 +
        (g[2:, 1:-1] - 2*g[1:-1, 1:-1] + g[:-2, 1:-1]) / dy**2
    )

    # Gradient of S
    grad_S_x = np.zeros_like(S)
    grad_S_y = np.zeros_like(S)
    grad_S_x[:, 1:-1] = (S[:, 2:] - S[:, :-2]) / (2 * dx)
    grad_S_y[1:-1, :] = (S[2:, :] - S[:-2, :]) / (2 * dy)

    # Gradient of log(S)
    S_safe = np.maximum(S, 1e-10)
    log_S = np.log(S_safe)
    grad_logS_x = np.zeros_like(S)
    grad_logS_y = np.zeros_like(S)
    grad_logS_x[:, 1:-1] = (log_S[:, 2:] - log_S[:, :-2]) / (2 * dx)
    grad_logS_y[1:-1, :] = (log_S[2:, :] - log_S[:-2, :]) / (2 * dy)

    # Chemotaxis flux: g * grad(log(S))
    flux_x = g * grad_logS_x
    flux_y = g * grad_logS_y

    # Divergence of flux
    div_flux = np.zeros_like(g)
    div_flux[:, 1:-1] += (flux_x[:, 2:] - flux_x[:, :-2]) / (2 * dx)
    div_flux[1:-1, :] += (flux_y[2:, :] - flux_y[:-2, :]) / (2 * dy)

    # PDE: You need to manually implement the specific form here
    # Default: diffusion-chemotaxis
    # ∂g/∂t = α·Δg - β·∇·(g∇(ln S))
    # WARNING: This is a template - may not match your actual PDE!

    if len(params) >= 2:
        alpha = params[0]
        beta = params[1]
        dg_dt = alpha * laplacian_g - beta * div_flux
    else:
        alpha = params[0] if len(params) > 0 else 0.1
        dg_dt = alpha * laplacian_g

    # Forward Euler update
    g_next = g + dt * dg_dt

    return g_next
'''

        return GeneratedSolver(
            source_code=template.strip(),
            function_name='pde_update',
            description=pde_description,
            parameters=param_names,
            success=True,
            error=None
        )

    def generate_from_llm(self, pde_str: str, param_names: list[str]) -> GeneratedSolver:
        """
        Generate solver code using LLM

        Args:
            pde_str: PDE equation string
            param_names: List of parameter names

        Returns:
            GeneratedSolver object
        """
        if self.llm_client is None:
            return self.generate_from_template(pde_str, param_names)

        prompt = f"""Generate a Python function to numerically solve this PDE for one time step:

PDE: {pde_str}
Parameters: {param_names}

Requirements:
1. Function signature: def pde_update(g: np.ndarray, S: np.ndarray, dx: float, dy: float, dt: float, params: np.ndarray) -> np.ndarray
2. Use finite differences (central difference for gradients, 5-point stencil for Laplacian)
3. params array contains [{', '.join(param_names)}] in order
4. Return updated g field after one time step dt
5. Use numpy operations only (no scipy)
6. Handle boundary conditions at edges (use zero gradient or copy interior)

Generate ONLY the function code, no explanation."""

        try:
            # Call LLM (implementation depends on llm_client interface)
            response = self.llm_client.generate(prompt)

            # Extract function
            func_body, func_name = self.extract_function_from_response(response)

            if func_body and func_name:
                return GeneratedSolver(
                    source_code=func_body,
                    function_name=func_name,
                    description=pde_str,
                    parameters=param_names,
                    success=True,
                    error=None
                )
            else:
                return GeneratedSolver(
                    source_code="",
                    function_name="",
                    description=pde_str,
                    parameters=param_names,
                    success=False,
                    error="Failed to extract function from LLM response"
                )

        except Exception as e:
            return GeneratedSolver(
                source_code="",
                function_name="",
                description=pde_str,
                parameters=param_names,
                success=False,
                error=f"LLM generation failed: {str(e)}"
            )


class CodeGenPDESolver:
    """
    PDE solver using code generation for exotic forms

    This is the Tier 2 fallback when symbolic evaluation fails
    """

    def __init__(self, dx: float = 1.0, dy: float = 1.0, dt: float = 0.01,
                 timeout_seconds: int = 30, llm_client: Optional[Any] = None):
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.sandbox = CodeExecutionSandbox(timeout_seconds=timeout_seconds)
        self.generator = PDECodeGenerator(llm_client=llm_client)
        self.code_cache = {}  # Cache generated code

    def solve(self, pde_str: str, g_init: np.ndarray, S: np.ndarray,
              param_values: Dict[str, float], num_steps: int,
              use_cache: bool = True) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Solve PDE using generated code

        Args:
            pde_str: PDE equation string
            g_init: Initial condition (H, W)
            S: Signal field (H, W) or (H, W, T)
            param_values: Parameter values dict
            num_steps: Number of time steps
            use_cache: Whether to use cached generated code

        Returns:
            (solution, info): Solution array (H, W, T) and info dict
        """
        info = {'method': 'code_generation', 'generation_time': 0, 'execution_time': 0}

        # Check cache
        cache_key = (pde_str, tuple(sorted(param_values.keys())))
        if use_cache and cache_key in self.code_cache:
            generated = self.code_cache[cache_key]
            info['cache_hit'] = True
        else:
            # Generate code
            start_time = time.time()
            param_names = list(param_values.keys())
            generated = self.generator.generate_from_llm(pde_str, param_names)
            info['generation_time'] = time.time() - start_time
            info['cache_hit'] = False

            if not generated.success:
                info['error'] = generated.error
                return None, info

            # Cache it
            if use_cache:
                self.code_cache[cache_key] = generated

        # Execute in sandbox
        start_time = time.time()
        result, success, error = self.sandbox.execute(
            code=generated.source_code,
            function_name=generated.function_name,
            g_init=g_init,
            S=S,
            params=param_values,
            dx=self.dx,
            dy=self.dy,
            dt=self.dt,
            num_steps=num_steps
        )
        info['execution_time'] = time.time() - start_time

        if not success:
            info['error'] = error
            info['generated_code'] = generated.source_code
            return None, info

        info['success'] = True
        info['generated_code'] = generated.source_code
        info['parameters'] = param_values

        return result, info
