"""
Pure LLMSR-style PDE Solver

The LLM directly generates a complete pde_update() function.
No symbolic parsing, no hardcoded forms - pure code generation.

Key advantages:
- Supports ANY number of parameters (not just α, β, γ, K)
- No GT-biased priors in prompts
- Maximum flexibility
"""

import numpy as np
import multiprocessing
import ast
import re
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import time
import sys
from io import StringIO
import scipy


# Module-level function for multiprocessing (can be pickled)
def _execute_pde_in_subprocess(code: str, g_init: np.ndarray, S: np.ndarray,
                                params: np.ndarray, num_steps: int, dx: float, dy: float, dt: float,
                                result_queue: multiprocessing.Queue):
    """Execute generated PDE code in isolated subprocess"""
    try:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        # Restricted namespace with scipy support
        import scipy.ndimage
        namespace = {
            'np': np,
            'numpy': np,
            'scipy': scipy,
            '__builtins__': {
                'range': range, 'len': len, 'min': min, 'max': max,
                'abs': abs, 'sum': sum, 'enumerate': enumerate,
                'zip': zip, 'float': float, 'int': int,
                'True': True, 'False': False, 'None': None,
                '__import__': __import__,  # Allow imports for scipy.ndimage
            },
        }

        # Execute code to define function
        exec(code, namespace)
        pde_update = namespace['pde_update']

        # Time integration
        H, W = g_init.shape
        g_history = np.zeros((H, W, num_steps))
        g_history[:, :, 0] = g_init.copy()
        g_current = g_init.copy()

        for t in range(1, num_steps):
            S_current = S[:, :, min(t, S.shape[2]-1)] if S.ndim == 3 else S

            g_next = pde_update(g_current, S_current, dx, dy, dt, params)

            # Validate
            if not isinstance(g_next, np.ndarray):
                result_queue.put((None, False, f"Must return np.ndarray, got {type(g_next)}"))
                return

            if g_next.shape != g_current.shape:
                result_queue.put((None, False, f"Shape mismatch: {g_next.shape} != {g_current.shape}"))
                return

            if np.any(np.isnan(g_next)) or np.any(np.isinf(g_next)):
                result_queue.put((None, False, f"NaN/Inf at step {t}"))
                return

            g_next = np.maximum(g_next, 0)
            g_history[:, :, t] = g_next
            g_current = g_next

        sys.stdout = old_stdout
        sys.stderr = old_stderr
        result_queue.put((g_history, True, None))

    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        import traceback
        result_queue.put((None, False, f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"))


def _execute_pde_inline(code: str, g_init: np.ndarray, S: np.ndarray,
                        params: np.ndarray, num_steps: int, dx: float, dy: float, dt: float):
    """Non-multiprocess execution of pde_update with rollout (H,W,T)."""
    try:
        namespace = {
            'np': np,
            'numpy': np,
            'scipy': scipy,
            '__builtins__': {
                'range': range, 'len': len, 'min': min, 'max': max,
                'abs': abs, 'sum': sum, 'enumerate': enumerate,
                'zip': zip, 'float': float, 'int': int,
                'True': True, 'False': False, 'None': None,
                '__import__': __import__,
            },
        }
        exec(code, namespace)
        pde_update = namespace['pde_update']

        H, W = g_init.shape
        g_history = np.zeros((H, W, num_steps))
        g_history[:, :, 0] = g_init.copy()
        g_current = np.expand_dims(g_init.copy(), axis=2)
        for t in range(1, num_steps):
            S_current = S[:, :, min(t, S.shape[2]-1)] if S.ndim == 3 else S
            g_next = pde_update(g_current, np.expand_dims(S_current, axis=2), dx, dy, dt, params)
            if not isinstance(g_next, np.ndarray):
                return None, False, f"Must return np.ndarray, got {type(g_next)}"
            if g_next.shape != g_current.shape:
                return None, False, f"Shape mismatch: {g_next.shape} != {g_current.shape}"
            if np.any(np.isnan(g_next)) or np.any(np.isinf(g_next)):
                return None, False, f"NaN/Inf at step {t}"
            g_next = np.maximum(g_next, 0)
            g_history[:, :, t:t+1] = g_next
            g_current = g_next
        return g_history, True, None
    except Exception as e:
        import traceback
        return None, False, f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


def _execute_pde_dgdt_in_subprocess(code: str, g_series: np.ndarray, S: np.ndarray,
                                     params: np.ndarray, dx: float, dy: float, dt: float,
                                     result_queue: multiprocessing.Queue):
    """Fast evaluation path: compare predicted dg/dt to observed dg/dt for all time steps.

    Assumes the generated pde_update can accept full time series:
    - g_series shape: (H, W, T)
    - S may be (H, W) or (H, W, T)
    pde_update should return predicted dg/dt with shape (H, W, T-1) or (H, W, T).
    We construct a one-step-ahead predicted sequence for visualization.
    """
    try:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        import scipy.ndimage
        namespace = {
            'np': np,
            'numpy': np,
            'scipy': scipy,
            '__builtins__': {
                'range': range, 'len': len, 'min': min, 'max': max,
                'abs': abs, 'sum': sum, 'enumerate': enumerate,
                'zip': zip, 'float': float, 'int': int,
                'True': True, 'False': False, 'None': None,
                '__import__': __import__,
            },
        }

        # Execute code to define function
        exec(code, namespace)
        pde_update = namespace['pde_update']

        if g_series.ndim != 3:
            result_queue.put((None, None, False, "g_series must be 3D (H,W,T) for fast evaluation"))
            return

        H, W, T = g_series.shape
        if T < 2:
            result_queue.put((None, None, False, "Not enough time steps for dg/dt comparison"))
            return

        # Compute observed dg/dt using forward difference
        dgdt_obs = (g_series[:, :, 1:] - g_series[:, :, :-1]) / dt
        if len(S.shape) < len(g_series.shape):
            S = np.repeat(np.expand_dims(S, axis=2), g_series.shape[2], axis=2)
        # Call model to get predicted dg/dt over all time steps
        try:
            dgdt_pred = pde_update(g_series, S, dx, dy, dt, params)
        except Exception as e:
            result_queue.put((None, None, False, f"pde_update call failed: {type(e).__name__}: {e}"))
            return

        if not isinstance(dgdt_pred, np.ndarray):
            result_queue.put((None, None, False, f"pde_update must return np.ndarray, got {type(dgdt_pred)}"))
            return

        if dgdt_pred.ndim != 3:
            result_queue.put((None, None, False, f"pde_update must return 3D array (H,W,T or T-1), got shape {dgdt_pred.shape}"))
            return

        # Align time dimension
        Tp = dgdt_pred.shape[2]
        if Tp == T:
            # Use first T-1 steps to match forward diff
            dgdt_pred_use = dgdt_pred[:, :, :-1]
        elif Tp == T - 1:
            dgdt_pred_use = dgdt_pred
        else:
            result_queue.put((None, None, False, f"Time dimension mismatch: pred T={Tp}, obs T={T-1}"))
            return

        # Build one-step-ahead predicted sequence for visualization
        g_pred = np.zeros_like(g_series)
        g_pred[:, :, 0] = g_series[:, :, 0]
        g_pred[:, :, 1:] = np.maximum(g_series[:, :, :-1] + dt * dgdt_pred_use, 0)

        sys.stdout = old_stdout
        sys.stderr = old_stderr
        result_queue.put((g_pred, dgdt_pred_use, True, None))
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        import traceback
        result_queue.put((None, None, False, f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"))


def _execute_pde_dgdt_inline(code: str, g_series: np.ndarray, S: np.ndarray,
                             params: np.ndarray, dx: float, dy: float, dt: float):
    """Non-multiprocess: compute dg/dt prediction for all time steps and build one-step preds."""
    try:
        namespace = {
            'np': np,
            'numpy': np,
            'scipy': scipy,
            '__builtins__': {
                'range': range, 'len': len, 'min': min, 'max': max,
                'abs': abs, 'sum': sum, 'enumerate': enumerate,
                'zip': zip, 'float': float, 'int': int,
                'True': True, 'False': False, 'None': None,
                '__import__': __import__,
            },
        }
        exec(code, namespace)
        pde_update = namespace['pde_update']

        if g_series.ndim != 3:
            return None, None, False, "g_series must be 3D (H,W,T) for fast evaluation"
        H, W, T = g_series.shape
        if T < 2:
            return None, None, False, "Not enough time steps for dg/dt comparison"

        # Expand S to 3D if needed for compatibility
        if S.ndim == 2:
            S_use = np.repeat(S[:, :, None], T, axis=2)
        else:
            S_use = S
        
        dgdt_pred = pde_update(g_series, S_use, dx, dy, dt, params)
        if not isinstance(dgdt_pred, np.ndarray):
            return None, None, False, f"pde_update must return np.ndarray, got {type(dgdt_pred)}"
        if dgdt_pred.ndim != 3:
            return None, None, False, f"pde_update must return 3D array (H,W,T or T-1), got shape {dgdt_pred.shape}"

        Tp = dgdt_pred.shape[2]
        if Tp == T:
            dgdt_use = dgdt_pred[:, :, :-1]
        elif Tp == T - 1:
            dgdt_use = dgdt_pred
        else:
            return None, None, False, f"Time dimension mismatch: pred T={Tp}, obs T={T-1}"

        g_pred = np.zeros_like(g_series)
        g_pred[:, :, 0] = g_series[:, :, 0]
        g_pred[:, :, 1:] = np.maximum(g_series[:, :, :-1] + dt * dgdt_use, 0)
        return g_pred, dgdt_use, True, None
    except Exception as e:
        import traceback
        return None, None, False, f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


@dataclass
class LLMSRPDESolver:
    """Pure LLMSR approach: LLM generates complete update function"""

    def __init__(self, llm_client, dx=1.0, dy=1.0, dt=0.01, timeout=30):
        self.llm_client = llm_client
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.timeout = timeout
        self.code_cache = {}

    def generate_pde_code(self, pde_description: str, num_params: int) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate PDE update code using LLM

        Args:
            pde_description: Natural language or equation string
            num_params: Number of parameters to optimize

        Returns:
            (code, error_message)
        """

        prompt = f"""Generate a Python function to numerically solve a PDE for ONE time step using scipy operators.

TASK: Implement the PDE described as: {pde_description}

REQUIREMENTS:
1. Function signature MUST be:
   def pde_update(g: np.ndarray, S: np.ndarray, dx: float, dy: float, dt: float, params: np.ndarray) -> np.ndarray

2. Inputs:
   - g: Current density field, 2D array (H, W)
   - S: Signal/chemoattractant field, 2D array (H, W)
   - dx, dy: Spatial step sizes
   - dt: Time step
   - params: Array of {num_params} parameters to be optimized, e.g., params[0], params[1], ...

3. Output:
   - g_next: Updated density field after time dt, same shape as g

4. Implementation:
   - MUST import and use scipy.ndimage operators for spatial derivatives
   - Use scipy.ndimage.laplace() for Laplacian: laplacian_g = scipy.ndimage.laplace(g) / dx**2
   - Use scipy.ndimage.sobel() or scipy.ndimage.convolve() for gradients
   - Handle boundaries automatically (scipy handles this)
   - Forward Euler: g_next = g + dt * dg_dt
   - You can use numpy and scipy.ndimage

5. Parameters:
   - Extract from params array: p0 = params[0], p1 = params[1], etc.
   - Use parameters as coefficients in PDE terms

EXAMPLE TEMPLATE (modify for your specific PDE):

```python
def pde_update(g: np.ndarray, S: np.ndarray, dx: float, dy: float, dt: float, params: np.ndarray) -> np.ndarray:
    import numpy as np
    import scipy.ndimage

    H, W = g.shape

    # Extract parameters
    p0 = params[0]  # e.g., diffusion coefficient
    p1 = params[1]  # e.g., chemotaxis strength
    # ... add more as needed

    # Compute Laplacian using scipy
    laplacian_g = scipy.ndimage.laplace(g) / (dx**2)

    # Compute gradients using scipy
    grad_g_x = scipy.ndimage.sobel(g, axis=1) / (2*dx)
    grad_g_y = scipy.ndimage.sobel(g, axis=0) / (2*dy)

    grad_S_x = scipy.ndimage.sobel(S, axis=1) / (2*dx)
    grad_S_y = scipy.ndimage.sobel(S, axis=0) / (2*dy)

    # Compute flux/divergence terms (if needed)
    # For chemotaxis: flux = g * grad_S
    # divergence can be computed as: div_flux = scipy.ndimage.convolve(flux_x, ...) + ...
    # Or manually: flux_x_grad = scipy.ndimage.sobel(flux_x, axis=1) / (2*dx)
    #              flux_y_grad = scipy.ndimage.sobel(flux_y, axis=0) / (2*dy)
    #              divergence = flux_x_grad + flux_y_grad

    # Compute dg/dt according to your PDE
    dg_dt = p0 * laplacian_g  # example: just diffusion

    # Forward Euler update
    g_next = g + dt * dg_dt

    # Enforce non-negativity
    g_next = np.maximum(g_next, 0)

    return g_next
```

Generate ONLY the function code, no explanation or markdown formatting."""

        try:
            response = self.llm_client.generate(prompt)

            # Extract function from response
            func_code, func_name = self._extract_function(response)

            if func_code and func_name == 'pde_update':
                return func_code, None
            else:
                return None, "Failed to extract pde_update function from LLM response"

        except Exception as e:
            return None, f"LLM generation failed: {str(e)}"

    def _extract_function(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract function code from LLM response"""
        # Remove markdown code blocks
        response = re.sub(r'```python\s*', '', response)
        response = re.sub(r'```\s*', '', response)

        # Find def pde_update
        pattern = r'def\s+pde_update\s*\([^)]*\)\s*(?:->.*?)?:'
        match = re.search(pattern, response)

        if not match:
            return None, None

        start_idx = match.start()
        code_from_def = response[start_idx:]

        # Try to parse with AST
        try:
            tree = ast.parse(code_from_def)
            if tree.body and isinstance(tree.body[0], ast.FunctionDef):
                func_node = tree.body[0]
                lines = code_from_def.split('\n')
                end_line = max(node.end_lineno for node in ast.walk(func_node) if hasattr(node, 'end_lineno'))
                function_code = '\n'.join(lines[:end_line])
                return function_code, 'pde_update'
        except:
            pass

        # Fallback: extract until dedent
        lines = code_from_def.split('\n')
        func_lines = [lines[0]]

        for line in lines[1:]:
            if line and not line[0].isspace() and line.strip():
                break
            func_lines.append(line)

        return '\n'.join(func_lines), 'pde_update'

    def evaluate_pde(self, code: str, g_init: np.ndarray, S: np.ndarray,
                     params: np.ndarray, num_steps: int) -> Tuple[Optional[np.ndarray], bool, Optional[str]]:
        """
        Evaluate generated PDE code inline (non-multiprocess).

        Returns: (solution, success, error_message)
        """
        return _execute_pde_inline(code, g_init, S, params, num_steps, self.dx, self.dy, self.dt)

    def evaluate_pde_dgdt(self, code: str, g_series: np.ndarray, S: np.ndarray,
                           params: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, Optional[str]]:
        """
        Fast evaluation inline: compare predicted dg/dt to observed dg/dt across all time steps.

        Returns:
            (g_pred, dgdt_pred_use, success, error)
            - g_pred: one-step-ahead sequence for visualization (H,W,T)
            - dgdt_pred_use: predicted dg/dt aligned to (H,W,T-1)
        """
        return _execute_pde_dgdt_inline(code, g_series, S, params, self.dx, self.dy, self.dt)

    def fit_and_evaluate(self, pde_description: str, num_params: int,
                        g_init: np.ndarray, S: np.ndarray, g_observed: np.ndarray,
                        param_bounds: List[Tuple[float, float]]) -> Dict:
        """
        Generate code, fit parameters, evaluate

        Returns dict with solution, fitted_params, metrics, etc.
        """
        from scipy import optimize

        # Check cache
        cache_key = (pde_description, num_params)
        if cache_key in self.code_cache:
            code = self.code_cache[cache_key]
        else:
            code, error = self.generate_pde_code(pde_description, num_params)
            if error:
                return {'success': False, 'error': error}
            self.code_cache[cache_key] = code

        num_steps = g_observed.shape[2]

        # Fit parameters
        def objective(params):
            solution, success, error = self.evaluate_pde(code, g_init, S, params, num_steps)
            if not success:
                return 1e10
            mse = np.mean((solution - g_observed) ** 2)
            return mse

        # Initial guess: midpoint of bounds
        x0 = [(b[0] + b[1]) / 2 for b in param_bounds]

        try:
            result = optimize.minimize(objective, x0, method='L-BFGS-B', bounds=param_bounds)
            fitted_params = result.x
            loss = result.fun
        except:
            fitted_params = np.array(x0)
            loss = 1e10

        # Final evaluation with fitted params
        solution, success, error = self.evaluate_pde(code, g_init, S, fitted_params, num_steps)

        if not success:
            return {'success': False, 'error': error, 'generated_code': code}

        # Compute metrics
        mse = float(np.mean((solution - g_observed) ** 2))
        ss_res = np.sum((g_observed - solution) ** 2)
        ss_tot = np.sum((g_observed - np.mean(g_observed)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-10))

        obs_mass = np.sum(g_observed, axis=(0, 1))
        pred_mass = np.sum(solution, axis=(0, 1))
        mass_error = float(np.abs(pred_mass[-1] - obs_mass[-1]) / obs_mass[-1] * 100)

        return {
            'success': True,
            'solution': solution,
            'fitted_params': fitted_params.tolist(),
            'mse': mse,
            'r2': r2,
            'mass_error': mass_error,
            'generated_code': code,
            'loss': loss
        }
