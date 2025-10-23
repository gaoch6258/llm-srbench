"""
PDE Solver for Spatiotemporal Equation Discovery

This module provides functionality to:
1. Parse symbolic PDE strings with operators ∇, ∇·, Δ, ∂/∂t
2. Solve 2D+time PDEs using finite difference methods
3. Compute spatiotemporal loss between predicted and observed fields
4. Fit constants in PDE templates using optimization
"""

import numpy as np
from scipy import optimize
from scipy.ndimage import laplace, convolve
import sympy as sp
from typing import Dict, Tuple, Optional, Callable, List
import re
from dataclasses import dataclass


@dataclass
class PDEConfig:
    """Configuration for PDE solver"""
    dx: float = 1.0  # Spatial step size in x
    dy: float = 1.0  # Spatial step size in y
    dt: float = 0.01  # Time step
    boundary_condition: str = "periodic"  # 'periodic', 'neumann', 'dirichlet'
    max_iterations: int = 1000  # Max time steps
    stability_check: bool = True  # Check CFL condition
    diffusion_limit: float = 0.5  # CFL stability limit for diffusion


class PDESolver:
    """Solver for 2D+time PDEs with chemotaxis and diffusion terms"""

    def __init__(self, config: Optional[PDEConfig] = None):
        self.config = config or PDEConfig()

    def gradient_2d(self, field: np.ndarray, dx: float = None, dy: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 2D gradient using central differences

        Args:
            field: 2D array (H, W)
            dx, dy: Spatial step sizes

        Returns:
            (grad_x, grad_y): Gradient components
        """
        dx = dx or self.config.dx
        dy = dy or self.config.dy

        # Central differences with appropriate boundary handling
        if self.config.boundary_condition == "periodic":
            grad_x = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2 * dx)
            grad_y = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2 * dy)
        else:
            grad_x = np.zeros_like(field)
            grad_y = np.zeros_like(field)

            # Interior points
            grad_x[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * dx)
            grad_y[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * dy)

            # Boundaries (one-sided differences)
            grad_x[:, 0] = (field[:, 1] - field[:, 0]) / dx
            grad_x[:, -1] = (field[:, -1] - field[:, -2]) / dx
            grad_y[0, :] = (field[1, :] - field[0, :]) / dy
            grad_y[-1, :] = (field[-1, :] - field[-2, :]) / dy

        return grad_x, grad_y

    def divergence_2d(self, field_x: np.ndarray, field_y: np.ndarray,
                      dx: float = None, dy: float = None) -> np.ndarray:
        """
        Compute 2D divergence of vector field

        Args:
            field_x, field_y: Vector field components (H, W)
            dx, dy: Spatial step sizes

        Returns:
            Divergence: scalar field (H, W)
        """
        dx = dx or self.config.dx
        dy = dy or self.config.dy

        if self.config.boundary_condition == "periodic":
            div_x = (np.roll(field_x, -1, axis=1) - np.roll(field_x, 1, axis=1)) / (2 * dx)
            div_y = (np.roll(field_y, -1, axis=0) - np.roll(field_y, 1, axis=0)) / (2 * dy)
        else:
            div_x = np.zeros_like(field_x)
            div_y = np.zeros_like(field_y)

            # Interior
            div_x[:, 1:-1] = (field_x[:, 2:] - field_x[:, :-2]) / (2 * dx)
            div_y[1:-1, :] = (field_y[2:, :] - field_y[:-2, :]) / (2 * dy)

            # Boundaries
            div_x[:, 0] = (field_x[:, 1] - field_x[:, 0]) / dx
            div_x[:, -1] = (field_x[:, -1] - field_x[:, -2]) / dx
            div_y[0, :] = (field_y[1, :] - field_y[0, :]) / dy
            div_y[-1, :] = (field_y[-1, :] - field_y[-2, :]) / dy

        return div_x + div_y

    def laplacian_2d(self, field: np.ndarray, dx: float = None, dy: float = None) -> np.ndarray:
        """
        Compute 2D Laplacian using 5-point stencil

        Args:
            field: 2D array (H, W)
            dx, dy: Spatial step sizes

        Returns:
            Laplacian: 2D array (H, W)
        """
        dx = dx or self.config.dx
        dy = dy or self.config.dy

        if self.config.boundary_condition == "periodic":
            # Use scipy's laplace with wrap mode for periodic BC
            laplacian = laplace(field, mode='wrap')
        else:
            laplacian = np.zeros_like(field)

            # 5-point stencil for interior
            laplacian[1:-1, 1:-1] = (
                (field[1:-1, 2:] - 2*field[1:-1, 1:-1] + field[1:-1, :-2]) / dx**2 +
                (field[2:, 1:-1] - 2*field[1:-1, 1:-1] + field[:-2, 1:-1]) / dy**2
            )

            # Neumann BC: zero-gradient at boundaries
            if self.config.boundary_condition == "neumann":
                # Copy interior values to boundaries
                laplacian[0, :] = laplacian[1, :]
                laplacian[-1, :] = laplacian[-2, :]
                laplacian[:, 0] = laplacian[:, 1]
                laplacian[:, -1] = laplacian[:, -2]
            # Dirichlet BC: zero at boundaries (already initialized to zero)

        return laplacian

    def time_derivative(self, field_current: np.ndarray, field_prev: np.ndarray,
                       dt: float = None) -> np.ndarray:
        """
        Compute time derivative using backward difference

        Args:
            field_current: Field at time t
            field_prev: Field at time t-dt
            dt: Time step

        Returns:
            Time derivative
        """
        dt = dt or self.config.dt
        return (field_current - field_prev) / dt

    def chemotaxis_term(self, g: np.ndarray, S: np.ndarray) -> np.ndarray:
        """
        Compute chemotaxis term: ∇·(g∇(ln S))

        This represents neutrophil migration up chemoattractant gradient.

        Args:
            g: Cell density field (H, W)
            S: Chemoattractant field (H, W)

        Returns:
            Chemotaxis term (H, W)
        """
        # Avoid log(0) by adding small epsilon
        eps = 1e-10
        S_safe = np.maximum(S, eps)

        # Compute ∇(ln S)
        grad_lnS_x, grad_lnS_y = self.gradient_2d(np.log(S_safe))

        # Compute g∇(ln S)
        flux_x = g * grad_lnS_x
        flux_y = g * grad_lnS_y

        # Compute ∇·(g∇(ln S))
        chemotaxis = self.divergence_2d(flux_x, flux_y)

        return chemotaxis

    def check_cfl_condition(self, alpha: float) -> bool:
        """
        Check CFL stability condition for explicit diffusion

        For 2D diffusion: dt <= dx^2 * dy^2 / (2 * alpha * (dx^2 + dy^2))

        Args:
            alpha: Diffusion coefficient

        Returns:
            True if stable, False otherwise
        """
        dx, dy, dt = self.config.dx, self.config.dy, self.config.dt
        max_dt = self.config.diffusion_limit * min(dx**2, dy**2) / (2 * abs(alpha))
        return dt <= max_dt

    def solve_reference_pde(self, g_init: np.ndarray, S: np.ndarray,
                           alpha: float, num_steps: int = None) -> np.ndarray:
        """
        Solve the reference chemotaxis PDE:
        ∂g/∂t = α·Δg - ∇·(g∇(ln S))

        Args:
            g_init: Initial cell density (H, W)
            S: Chemoattractant field (H, W, T) or (H, W) if static
            alpha: Diffusion coefficient
            num_steps: Number of time steps (default: config.max_iterations)

        Returns:
            g_history: Cell density evolution (H, W, T)
        """
        if self.config.stability_check and not self.check_cfl_condition(alpha):
            print(f"Warning: CFL condition violated with α={alpha}, dt={self.config.dt}")

        num_steps = num_steps or self.config.max_iterations
        H, W = g_init.shape
        g_history = np.zeros((H, W, num_steps))
        g_history[:, :, 0] = g_init.copy()

        g_current = g_init.copy()

        for t in range(1, num_steps):
            # Get current S field (may be time-varying or static)
            if S.ndim == 3:
                S_current = S[:, :, min(t, S.shape[2]-1)]
            else:
                S_current = S

            # Compute diffusion term: α·Δg
            diffusion = alpha * self.laplacian_2d(g_current)

            # Compute chemotaxis term: -∇·(g∇(ln S))
            chemotaxis = -self.chemotaxis_term(g_current, S_current)

            # Update: g^{n+1} = g^n + dt * (α·Δg - ∇·(g∇(ln S)))
            g_next = g_current + self.config.dt * (diffusion + chemotaxis)

            # Ensure non-negativity (physical constraint)
            g_next = np.maximum(g_next, 0)

            g_history[:, :, t] = g_next
            g_current = g_next

        return g_history

    def parse_pde_string(self, pde_str: str) -> Dict:
        """
        Parse symbolic PDE string into components

        Supported operators:
        - ∇ or grad: gradient
        - ∇· or div: divergence
        - Δ or laplacian: Laplacian
        - ∂/∂t or dt: time derivative

        Args:
            pde_str: PDE equation string

        Returns:
            Dictionary with parsed components and structure
        """
        # Normalize operators
        pde_normalized = pde_str.replace('∇·', 'div')
        pde_normalized = pde_normalized.replace('∇', 'grad')
        pde_normalized = pde_normalized.replace('Δ', 'laplacian')
        pde_normalized = pde_normalized.replace('∂/∂t', 'dt')

        # Detect operators
        has_gradient = 'grad' in pde_normalized
        has_divergence = 'div' in pde_normalized
        has_laplacian = 'laplacian' in pde_normalized
        has_time_deriv = 'dt' in pde_normalized

        # Extract parameters (e.g., α, β, etc.)
        param_pattern = r'([α-ωa-zA-Z]\w*)'
        potential_params = re.findall(param_pattern, pde_str)

        # Filter out known variables (g, S, x, y, t)
        known_vars = {'g', 'S', 'x', 'y', 't', 'grad', 'div', 'laplacian', 'dt', 'ln', 'log', 'exp', 'sin', 'cos'}
        parameters = [p for p in potential_params if p not in known_vars]

        return {
            'original': pde_str,
            'normalized': pde_normalized,
            'has_gradient': has_gradient,
            'has_divergence': has_divergence,
            'has_laplacian': has_laplacian,
            'has_time_derivative': has_time_deriv,
            'parameters': list(set(parameters)),
            'is_pde': has_gradient or has_divergence or has_laplacian or has_time_deriv
        }

    def evaluate_pde(self, pde_str: str, g_init: np.ndarray, S: np.ndarray,
                     param_values: Dict[str, float] = None,
                     num_steps: int = None) -> Tuple[np.ndarray, Dict]:
        """
        Evaluate a PDE string by solving it numerically

        Args:
            pde_str: PDE equation string
            g_init: Initial condition (H, W)
            S: Chemoattractant field (H, W) or (H, W, T)
            param_values: Dictionary of parameter values (e.g., {'α': 0.1})
            num_steps: Number of time steps

        Returns:
            (solution, info): Solution array (H, W, T) and info dict
        """
        parsed = self.parse_pde_string(pde_str)

        if not parsed['is_pde']:
            raise ValueError(f"String does not appear to be a PDE: {pde_str}")

        # Default parameters
        param_values = param_values or {}

        # For now, implement specific PDEs
        # This can be extended to a more general symbolic parser

        # Check if it's the reference chemotaxis PDE
        if 'div' in parsed['normalized'] and 'laplacian' in parsed['normalized']:
            # Assume form: ∂g/∂t = α·Δg - ∇·(g∇(ln S))
            alpha = param_values.get('α', param_values.get('alpha', 0.1))
            solution = self.solve_reference_pde(g_init, S, alpha, num_steps)
            info = {'alpha': alpha, 'type': 'chemotaxis'}

        elif 'laplacian' in parsed['normalized'] and not 'div' in parsed['normalized']:
            # Pure diffusion: ∂g/∂t = α·Δg
            alpha = param_values.get('α', param_values.get('alpha', 0.1))
            solution = self.solve_diffusion(g_init, alpha, num_steps)
            info = {'alpha': alpha, 'type': 'diffusion'}

        else:
            # Generic PDE - would need more sophisticated parsing
            raise NotImplementedError(f"PDE form not yet supported: {pde_str}")

        return solution, info

    def solve_diffusion(self, g_init: np.ndarray, alpha: float,
                       num_steps: int = None) -> np.ndarray:
        """
        Solve pure diffusion PDE: ∂g/∂t = α·Δg

        Args:
            g_init: Initial condition (H, W)
            alpha: Diffusion coefficient
            num_steps: Number of time steps

        Returns:
            g_history: Evolution (H, W, T)
        """
        if self.config.stability_check and not self.check_cfl_condition(alpha):
            print(f"Warning: CFL condition violated with α={alpha}")

        num_steps = num_steps or self.config.max_iterations
        H, W = g_init.shape
        g_history = np.zeros((H, W, num_steps))
        g_history[:, :, 0] = g_init.copy()

        g_current = g_init.copy()

        for t in range(1, num_steps):
            laplacian = self.laplacian_2d(g_current)
            g_next = g_current + self.config.dt * alpha * laplacian
            g_next = np.maximum(g_next, 0)

            g_history[:, :, t] = g_next
            g_current = g_next

        return g_history

    def compute_spatiotemporal_loss(self, predicted: np.ndarray,
                                   observed: np.ndarray,
                                   metric: str = 'mse') -> float:
        """
        Compute loss between predicted and observed spatiotemporal fields

        Args:
            predicted: Predicted field (H, W, T)
            observed: Observed field (H, W, T)
            metric: Loss metric ('mse', 'rmse', 'nmse', 'r2')

        Returns:
            Loss value
        """
        if predicted.shape != observed.shape:
            # Interpolate or truncate to match
            min_T = min(predicted.shape[2], observed.shape[2])
            predicted = predicted[:, :, :min_T]
            observed = observed[:, :, :min_T]

        if metric == 'mse':
            return np.mean((predicted - observed) ** 2)

        elif metric == 'rmse':
            return np.sqrt(np.mean((predicted - observed) ** 2))

        elif metric == 'nmse':
            mse = np.mean((predicted - observed) ** 2)
            variance = np.var(observed)
            return mse / (variance + 1e-10)

        elif metric == 'r2':
            ss_res = np.sum((observed - predicted) ** 2)
            ss_tot = np.sum((observed - np.mean(observed)) ** 2)
            return 1 - ss_res / (ss_tot + 1e-10)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def fit_pde_parameters(self, pde_template: str, g_init: np.ndarray,
                          S: np.ndarray, g_observed: np.ndarray,
                          param_bounds: Dict[str, Tuple[float, float]] = None,
                          method: str = 'L-BFGS-B') -> Tuple[Dict[str, float], float]:
        """
        Fit parameters in a PDE template to observed data

        Args:
            pde_template: PDE string with parameters to fit
            g_init: Initial condition
            S: Chemoattractant field
            g_observed: Observed evolution (H, W, T)
            param_bounds: Parameter bounds dict, e.g., {'α': (0, 1)}
            method: Optimization method

        Returns:
            (best_params, best_loss): Optimized parameters and loss
        """
        parsed = self.parse_pde_string(pde_template)
        param_names = parsed['parameters']

        if not param_names:
            raise ValueError("No parameters found in PDE template")

        # Default bounds
        param_bounds = param_bounds or {}
        bounds = [param_bounds.get(p, (0.001, 10.0)) for p in param_names]

        # Initial guess (midpoint of bounds)
        x0 = [(b[0] + b[1]) / 2 for b in bounds]

        num_steps = g_observed.shape[2]

        def objective(params):
            """Objective function for optimization"""
            param_dict = {name: val for name, val in zip(param_names, params)}

            try:
                predicted, _ = self.evaluate_pde(pde_template, g_init, S,
                                                 param_dict, num_steps)
                loss = self.compute_spatiotemporal_loss(predicted, g_observed, 'mse')
                return loss
            except Exception as e:
                # Return large penalty for invalid parameters
                return 1e10

        # Optimize
        result = optimize.minimize(objective, x0, method=method, bounds=bounds)

        best_params = {name: val for name, val in zip(param_names, result.x)}
        best_loss = result.fun

        return best_params, best_loss


def detect_equation_type(equation_str: str) -> str:
    """
    Detect if equation is a PDE or algebraic expression

    Args:
        equation_str: Equation string

    Returns:
        'pde' or 'algebraic'
    """
    pde_indicators = ['∇', '∇·', 'Δ', 'grad', 'div', 'laplacian', '∂', 'dt']

    for indicator in pde_indicators:
        if indicator in equation_str:
            return 'pde'

    return 'algebraic'


def create_chemotaxis_datamodule(data_path: str = None) -> Dict:
    """
    Create a synthetic chemotaxis dataset for testing

    Returns:
        Dictionary with g_init, S, g_observed, metadata
    """
    # Create synthetic data matching the problem definition
    H, W = 256, 256
    T = 100

    # Initial cell density (Gaussian blob)
    x = np.linspace(0, H-1, H)
    y = np.linspace(0, W-1, W)
    X, Y = np.meshgrid(x, y)

    # Start with cells concentrated in center
    g_init = np.exp(-((X - H/2)**2 + (Y - W/2)**2) / (0.1 * H**2))

    # Chemoattractant field (gradient pointing to upper-right)
    # S(x,y) = exp(0.01*x + 0.01*y)
    S = np.exp(0.01 * X + 0.01 * Y)

    # Generate ground truth evolution using reference PDE
    solver = PDESolver(PDEConfig(dx=1.0, dy=1.0, dt=0.01))
    alpha_true = 0.5
    g_observed = solver.solve_reference_pde(g_init, S, alpha_true, num_steps=T)

    return {
        'g_init': g_init,
        'S': S,
        'g_observed': g_observed,
        'alpha_true': alpha_true,
        'metadata': {
            'shape': (H, W, T),
            'reference_pde': '∇·(g∇(ln S)) = α·Δg - ∂g/∂t',
            'dx': 1.0,
            'dy': 1.0,
            'dt': 0.01
        }
    }
