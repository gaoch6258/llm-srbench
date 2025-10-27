"""
Symbolic PDE Solver using SymPy for General PDE Forms

This module provides symbolic parsing and automatic code generation
for a wide range of PDE forms beyond the hardcoded patterns.

Architecture:
- Parse PDE string using SymPy
- Extract spatial operators (Laplacian, Divergence, Gradient)
- Generate finite difference code automatically
- Compile and execute with JIT optimization
"""

import numpy as np
import sympy as sp
from sympy import symbols, Function, Derivative, sympify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from typing import Dict, Tuple, Optional, List, Set, Any
import re
from dataclasses import dataclass
from enum import Enum


class OperatorType(Enum):
    """Types of differential operators"""
    LAPLACIAN = "laplacian"
    DIVERGENCE = "divergence"
    GRADIENT = "gradient"
    TIME_DERIVATIVE = "time_derivative"


@dataclass
class SpatialOperator:
    """Represents a spatial differential operator"""
    op_type: OperatorType
    argument: sp.Expr  # The expression being operated on
    coefficient: sp.Expr = sp.S.One  # Multiplicative coefficient


@dataclass
class PDEStructure:
    """Structured representation of a parsed PDE"""
    time_derivative: Optional[sp.Expr]  # LHS: ∂g/∂t
    spatial_operators: List[SpatialOperator]  # Spatial terms
    reaction_terms: List[sp.Expr]  # Non-differential terms
    parameters: Set[str]  # Parameter names (α, β, etc.)
    variables: Dict[str, str]  # Variable names and their meanings
    is_valid: bool = True
    error_message: Optional[str] = None


class SymbolicPDEParser:
    """Parse PDE strings using SymPy for maximum flexibility"""

    def __init__(self):
        # Define standard symbols
        self.g = sp.Symbol('g', real=True, positive=True)  # Density field
        self.S = sp.Symbol('S', real=True, positive=True)  # Signal field
        self.x, self.y, self.t = symbols('x y t', real=True)

        # Standard operators as functions
        self.grad = sp.Function('grad')
        self.div = sp.Function('div')
        self.laplacian = sp.Function('laplacian')

        # Greek letters commonly used as parameters
        self.greek_params = {'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'κ', 'λ', 'μ', 'ν', 'ξ', 'π', 'ρ', 'σ', 'τ', 'φ', 'χ', 'ψ', 'ω'}

    def normalize_equation(self, pde_str: str) -> str:
        """
        Normalize PDE string to SymPy-compatible format

        Args:
            pde_str: Raw PDE string (e.g., "∂g/∂t = α·Δg - β·∇·(g∇S)")

        Returns:
            Normalized string for SymPy parsing
        """
        normalized = pde_str

        # Replace Unicode operators with function names BEFORE replacing ·
        # This prevents Δg becoming separate letters
        normalized = normalized.replace('∂/∂t', 'dt')
        normalized = normalized.replace('∂g/∂t', 'dt(g)')
        normalized = normalized.replace('Δg', 'laplacian(g)')
        normalized = normalized.replace('Δ', 'laplacian')
        normalized = normalized.replace('∇·', 'div')
        normalized = normalized.replace('∇', 'grad')

        # Handle ln(S) -> log(S) BEFORE replacing ·
        normalized = normalized.replace('ln(', 'log(')

        # Replace multiplication symbols LAST
        normalized = normalized.replace('·', '*')
        normalized = normalized.replace('×', '*')

        return normalized

    def extract_operators(self, expr: sp.Expr) -> List[SpatialOperator]:
        """
        Extract spatial operators from SymPy expression

        Args:
            expr: SymPy expression

        Returns:
            List of SpatialOperator objects
        """
        operators = []

        # Traverse expression tree
        for term in sp.Add.make_args(expr):
            # Extract coefficient
            coeff = sp.S.One
            core = term

            if term.is_Mul:
                args = sp.Mul.make_args(term)
                coeff_parts = []
                func_parts = []

                for arg in args:
                    if arg.is_Function or arg.has(sp.Function):
                        func_parts.append(arg)
                    else:
                        coeff_parts.append(arg)

                coeff = sp.Mul(*coeff_parts) if coeff_parts else sp.S.One
                core = sp.Mul(*func_parts) if func_parts else term

            # Identify operator type
            if core.has(self.laplacian):
                # Find laplacian(...)
                for func_call in sp.preorder_traversal(core):
                    if isinstance(func_call, sp.Function) and func_call.func.__name__ == 'laplacian':
                        arg = func_call.args[0] if func_call.args else self.g
                        operators.append(SpatialOperator(
                            op_type=OperatorType.LAPLACIAN,
                            argument=arg,
                            coefficient=coeff
                        ))
                        break

            elif core.has(self.div):
                # Find div(...)
                for func_call in sp.preorder_traversal(core):
                    if isinstance(func_call, sp.Function) and func_call.func.__name__ == 'div':
                        arg = func_call.args[0] if func_call.args else self.g
                        operators.append(SpatialOperator(
                            op_type=OperatorType.DIVERGENCE,
                            argument=arg,
                            coefficient=coeff
                        ))
                        break

            elif core.has(self.grad):
                # Find grad(...)
                for func_call in sp.preorder_traversal(core):
                    if isinstance(func_call, sp.Function) and func_call.func.__name__ == 'grad':
                        arg = func_call.args[0] if func_call.args else self.S
                        operators.append(SpatialOperator(
                            op_type=OperatorType.GRADIENT,
                            argument=arg,
                            coefficient=coeff
                        ))
                        break

        return operators

    def extract_reaction_terms(self, expr: sp.Expr) -> List[sp.Expr]:
        """
        Extract non-differential (reaction) terms

        Args:
            expr: SymPy expression

        Returns:
            List of reaction term expressions
        """
        reaction_terms = []

        for term in sp.Add.make_args(expr):
            # If term doesn't contain spatial operators, it's a reaction term
            has_spatial_op = (
                term.has(self.laplacian) or
                term.has(self.div) or
                term.has(self.grad)
            )

            if not has_spatial_op:
                reaction_terms.append(term)

        return reaction_terms

    def extract_parameters(self, expr: sp.Expr, exclude_vars: Set[str] = None) -> Set[str]:
        """
        Extract parameter names from expression

        Args:
            expr: SymPy expression
            exclude_vars: Variable names to exclude (g, S, x, y, t)

        Returns:
            Set of parameter names
        """
        exclude_vars = exclude_vars or {'g', 'S', 'x', 'y', 't', 'e', 'E', 'I', 'N', 'O', 'Q'}

        # Get all symbols
        all_symbols = expr.free_symbols

        # Filter to get parameters
        params = {
            str(sym) for sym in all_symbols
            if str(sym) not in exclude_vars and len(str(sym)) <= 3
        }

        return params

    def parse(self, pde_str: str) -> PDEStructure:
        """
        Parse PDE string into structured representation

        Args:
            pde_str: PDE equation string

        Returns:
            PDEStructure object with parsed components
        """
        try:
            # Normalize
            normalized = self.normalize_equation(pde_str)

            # Split on '=' to get LHS and RHS
            if '=' in normalized:
                lhs, rhs = normalized.split('=', 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
            else:
                # Assume format is just RHS (e.g., "α·Δg")
                lhs = 'dt(g)'
                rhs = normalized.strip()

            # Parse RHS with SymPy
            # Use local dict for custom functions
            local_dict = {
                'g': self.g,
                'S': self.S,
                'x': self.x,
                'y': self.y,
                't': self.t,
                'grad': self.grad,
                'div': self.div,
                'laplacian': self.laplacian,
                'dt': sp.Function('dt'),
                'log': sp.log,
                'ln': sp.log,
                'exp': sp.exp,
                'sin': sp.sin,
                'cos': sp.cos,
            }

            # Add Greek letters as symbols
            for greek in self.greek_params:
                if greek in rhs:
                    local_dict[greek] = sp.Symbol(greek, real=True)

            # Parse RHS
            try:
                rhs_expr = parse_expr(rhs, local_dict=local_dict,
                                     transformations=(standard_transformations +
                                                     (implicit_multiplication_application,)))
            except Exception as e:
                # Fallback: try sympify
                rhs_expr = sympify(rhs, locals=local_dict)

            # Extract components
            spatial_ops = self.extract_operators(rhs_expr)
            reaction_terms = self.extract_reaction_terms(rhs_expr)
            parameters = self.extract_parameters(rhs_expr)

            # Parse LHS (time derivative)
            time_deriv = None
            if 'dt' in lhs:
                time_deriv = self.g

            return PDEStructure(
                time_derivative=time_deriv,
                spatial_operators=spatial_ops,
                reaction_terms=reaction_terms,
                parameters=parameters,
                variables={'g': 'density', 'S': 'signal'},
                is_valid=True,
                error_message=None
            )

        except Exception as e:
            return PDEStructure(
                time_derivative=None,
                spatial_operators=[],
                reaction_terms=[],
                parameters=set(),
                variables={},
                is_valid=False,
                error_message=f"Parse error: {str(e)}"
            )

    def can_handle_symbolically(self, structure: PDEStructure) -> bool:
        """
        Determine if this PDE can be handled with symbolic finite differences

        Args:
            structure: Parsed PDE structure

        Returns:
            True if can be handled symbolically, False if need code generation
        """
        if not structure.is_valid:
            return False

        # Check for unsupported operators
        unsupported_patterns = [
            # No spatially-varying diffusion coefficients D(x, y)
            lambda op: (op.op_type == OperatorType.LAPLACIAN and
                       op.coefficient.has(self.x) or op.coefficient.has(self.y)),

            # No gradient of coordinates ∇x, ∇y
            lambda op: (op.op_type == OperatorType.GRADIENT and
                       (op.argument == self.x or op.argument == self.y)),
        ]

        for op in structure.spatial_operators:
            for pattern in unsupported_patterns:
                if pattern(op):
                    return False

        return True


class SymbolicPDEEvaluator:
    """Evaluate PDEs using symbolically-generated finite difference schemes"""

    def __init__(self, dx: float = 1.0, dy: float = 1.0, dt: float = 0.01,
                 boundary_condition: str = "periodic"):
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.boundary_condition = boundary_condition
        self.parser = SymbolicPDEParser()

    def laplacian_2d(self, field: np.ndarray) -> np.ndarray:
        """Compute 2D Laplacian using 5-point stencil"""
        if self.boundary_condition == "periodic":
            from scipy.ndimage import laplace
            return laplace(field, mode='wrap')
        else:
            laplacian = np.zeros_like(field)
            laplacian[1:-1, 1:-1] = (
                (field[1:-1, 2:] - 2*field[1:-1, 1:-1] + field[1:-1, :-2]) / self.dx**2 +
                (field[2:, 1:-1] - 2*field[1:-1, 1:-1] + field[:-2, 1:-1]) / self.dy**2
            )
            return laplacian

    def gradient_2d(self, field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 2D gradient using central differences"""
        if self.boundary_condition == "periodic":
            grad_x = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2 * self.dx)
            grad_y = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2 * self.dy)
        else:
            grad_x = np.zeros_like(field)
            grad_y = np.zeros_like(field)
            grad_x[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * self.dx)
            grad_y[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * self.dy)
        return grad_x, grad_y

    def divergence_2d(self, field_x: np.ndarray, field_y: np.ndarray) -> np.ndarray:
        """Compute 2D divergence"""
        if self.boundary_condition == "periodic":
            div_x = (np.roll(field_x, -1, axis=1) - np.roll(field_x, 1, axis=1)) / (2 * self.dx)
            div_y = (np.roll(field_y, -1, axis=0) - np.roll(field_y, 1, axis=0)) / (2 * self.dy)
        else:
            div_x = np.zeros_like(field_x)
            div_y = np.zeros_like(field_y)
            div_x[:, 1:-1] = (field_x[:, 2:] - field_x[:, :-2]) / (2 * self.dx)
            div_y[1:-1, :] = (field_y[2:, :] - field_y[:-2, :]) / (2 * self.dy)
        return div_x + div_y

    def evaluate_operator(self, op: SpatialOperator, g: np.ndarray,
                         S: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """
        Evaluate a spatial operator

        Args:
            op: SpatialOperator to evaluate
            g: Current density field
            S: Signal field
            params: Parameter values

        Returns:
            Evaluated operator result
        """
        # Evaluate coefficient - substitute parameters first
        try:
            coeff_expr = op.coefficient.subs(params)
            # If still has symbols, try to evaluate or default to 1.0
            if coeff_expr.free_symbols:
                # Try evalf
                coeff_val = float(coeff_expr.evalf())
            else:
                coeff_val = float(coeff_expr)
        except (TypeError, ValueError):
            # If conversion fails, default to 1.0
            coeff_val = 1.0

        # Get argument field
        arg_str = str(op.argument)
        if arg_str == 'g':
            arg_field = g
        elif arg_str == 'S':
            arg_field = S
        elif 'log' in arg_str or 'ln' in arg_str:
            # Handle log(S), ln(S)
            eps = 1e-10
            arg_field = np.log(np.maximum(S, eps))
        else:
            # Try to evaluate as expression
            # This handles cases like g*S, g**2, etc.
            local_vars = {'g': g, 'S': S, 'np': np}
            local_vars.update({k: v for k, v in params.items()})
            try:
                arg_field = eval(arg_str, {"__builtins__": {}}, local_vars)
            except:
                arg_field = g  # Fallback

        # Apply operator
        if op.op_type == OperatorType.LAPLACIAN:
            result = self.laplacian_2d(arg_field)

        elif op.op_type == OperatorType.GRADIENT:
            result_x, result_y = self.gradient_2d(arg_field)
            result = (result_x, result_y)  # Return tuple for gradient

        elif op.op_type == OperatorType.DIVERGENCE:
            # Divergence argument should be a vector field
            # Common pattern: div(g*grad(S)) or div(g*grad(log(S)))
            arg_str = str(op.argument)

            if 'grad' in arg_str:
                # Extract what's being grad'd
                if 'log(S)' in arg_str or 'ln(S)' in arg_str:
                    eps = 1e-10
                    S_safe = np.maximum(S, eps)
                    grad_x, grad_y = self.gradient_2d(np.log(S_safe))
                elif 'S' in arg_str:
                    grad_x, grad_y = self.gradient_2d(S)
                else:
                    grad_x, grad_y = self.gradient_2d(g)

                # Multiply by g if present
                if '*g' in arg_str or 'g*' in arg_str:
                    flux_x = g * grad_x
                    flux_y = g * grad_y
                else:
                    flux_x = grad_x
                    flux_y = grad_y

                result = self.divergence_2d(flux_x, flux_y)
            else:
                # Direct divergence (rare)
                result = self.divergence_2d(arg_field, arg_field)
        else:
            result = np.zeros_like(g)

        return coeff_val * result if not isinstance(result, tuple) else (coeff_val * result[0], coeff_val * result[1])

    def evaluate_reaction(self, term: sp.Expr, g: np.ndarray,
                         S: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Evaluate a reaction term"""
        # Substitute parameters
        term_with_params = term.subs(params)

        # Convert to numpy-evaluable expression
        term_str = str(term_with_params)

        # Create safe evaluation context
        local_vars = {
            'g': g,
            'S': S,
            'np': np,
            'exp': np.exp,
            'log': np.log,
            'sin': np.sin,
            'cos': np.cos,
        }

        try:
            result = eval(term_str, {"__builtins__": {}}, local_vars)
            return result
        except:
            return np.zeros_like(g)

    def solve(self, pde_str: str, g_init: np.ndarray, S: np.ndarray,
              param_values: Dict[str, float], num_steps: int) -> Tuple[np.ndarray, Dict]:
        """
        Solve PDE using symbolic finite differences

        Args:
            pde_str: PDE equation string
            g_init: Initial condition (H, W)
            S: Signal field (H, W) or (H, W, T)
            param_values: Parameter values dict
            num_steps: Number of time steps

        Returns:
            (solution, info): Solution array (H, W, T) and info dict
        """
        # Parse PDE
        structure = self.parser.parse(pde_str)

        if not structure.is_valid:
            raise ValueError(f"Invalid PDE: {structure.error_message}")

        if not self.parser.can_handle_symbolically(structure):
            raise NotImplementedError("PDE requires code generation (use hybrid tier 2)")

        # Initialize solution
        H, W = g_init.shape
        g_history = np.zeros((H, W, num_steps))
        g_history[:, :, 0] = g_init.copy()
        g_current = g_init.copy()

        # Time integration
        for t in range(1, num_steps):
            # Get current S field
            if S.ndim == 3:
                S_current = S[:, :, min(t, S.shape[2]-1)]
            else:
                S_current = S

            # Compute RHS: sum of all operators and reaction terms
            rhs = np.zeros_like(g_current)

            # Add spatial operators
            for op in structure.spatial_operators:
                op_result = self.evaluate_operator(op, g_current, S_current, param_values)
                if isinstance(op_result, tuple):
                    # Gradient returns tuple, skip for now (handled by divergence)
                    continue
                rhs += op_result

            # Add reaction terms
            for term in structure.reaction_terms:
                rhs += self.evaluate_reaction(term, g_current, S_current, param_values)

            # Update: forward Euler
            g_next = g_current + self.dt * rhs

            # Physical constraints
            g_next = np.maximum(g_next, 0)  # Non-negativity

            g_history[:, :, t] = g_next
            g_current = g_next

        info = {
            'method': 'symbolic_fd',
            'parameters': param_values,
            'structure': structure
        }

        return g_history, info
