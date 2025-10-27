"""
Hybrid PDE Solver - Intelligent Router

This module provides a unified interface that automatically routes PDE
evaluation to the most appropriate solver:

Tier 1 (Fast Path): Symbolic evaluation with SymPy + automatic finite differences
Tier 2 (Fallback): Code generation with sandbox execution

Decision Logic:
1. Try to parse with SymPy
2. Check if form is symbolically solvable
3. If yes -> use symbolic solver (fast, safe)
4. If no -> fall back to code generation (flexible, slower)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
from enum import Enum
import time

from .pde_symbolic_solver import SymbolicPDEParser, SymbolicPDEEvaluator, PDEStructure
from .pde_codegen_solver import CodeGenPDESolver


class SolverTier(Enum):
    """Which solver tier was used"""
    SYMBOLIC = "symbolic"
    CODEGEN = "codegen"
    LEGACY = "legacy"  # Original hardcoded patterns


@dataclass
class HybridSolverResult:
    """Result from hybrid solver"""
    solution: Optional[np.ndarray]  # (H, W, T)
    success: bool
    tier_used: SolverTier
    parse_info: Optional[PDEStructure]
    execution_time: float
    error_message: Optional[str] = None
    info: Optional[Dict] = None


class PDEHybridSolver:
    """
    Hybrid PDE solver with intelligent routing

    Automatically selects the best evaluation strategy:
    1. Symbolic (fast, safe, broad support)
    2. Code generation (flexible, exotic forms)
    3. Legacy (original hardcoded chemotaxis/diffusion)
    """

    def __init__(self,
                 dx: float = 1.0,
                 dy: float = 1.0,
                 dt: float = 0.01,
                 boundary_condition: str = "periodic",
                 codegen_timeout: int = 30,
                 llm_client: Optional[Any] = None,
                 prefer_symbolic: bool = True,
                 enable_codegen: bool = True,
                 verbose: bool = False):
        """
        Args:
            dx, dy, dt: Discretization parameters
            boundary_condition: "periodic" or "neumann"
            codegen_timeout: Timeout for code generation execution
            llm_client: Optional LLM client for code generation
            prefer_symbolic: Try symbolic solver first
            enable_codegen: Enable code generation fallback
            verbose: Print routing decisions
        """
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.boundary_condition = boundary_condition
        self.prefer_symbolic = prefer_symbolic
        self.enable_codegen = enable_codegen
        self.verbose = verbose

        # Initialize solvers
        self.parser = SymbolicPDEParser()
        self.symbolic_solver = SymbolicPDEEvaluator(
            dx=dx, dy=dy, dt=dt, boundary_condition=boundary_condition
        )
        self.codegen_solver = CodeGenPDESolver(
            dx=dx, dy=dy, dt=dt,
            timeout_seconds=codegen_timeout,
            llm_client=llm_client
        )

        # Statistics
        self.stats = {
            'symbolic_success': 0,
            'symbolic_failures': 0,
            'codegen_success': 0,
            'codegen_failures': 0,
            'legacy_success': 0,
            'parse_failures': 0,
        }

    def _try_symbolic_solve(self, pde_str: str, g_init: np.ndarray, S: np.ndarray,
                           param_values: Dict[str, float], num_steps: int) -> Tuple[Optional[np.ndarray], Optional[PDEStructure], Optional[str]]:
        """
        Attempt symbolic solution

        Returns:
            (solution, parse_info, error_message)
        """
        try:
            # Parse PDE
            structure = self.parser.parse(pde_str)

            if not structure.is_valid:
                return None, structure, f"Parse failed: {structure.error_message}"

            # Check if symbolically solvable
            if not self.parser.can_handle_symbolically(structure):
                return None, structure, "PDE form requires code generation (has unsupported patterns)"

            # Solve
            solution, info = self.symbolic_solver.solve(
                pde_str, g_init, S, param_values, num_steps
            )

            return solution, structure, None

        except Exception as e:
            return None, None, f"Symbolic solver error: {str(e)}"

    def _try_codegen_solve(self, pde_str: str, g_init: np.ndarray, S: np.ndarray,
                          param_values: Dict[str, float], num_steps: int) -> Tuple[Optional[np.ndarray], Optional[Dict], Optional[str]]:
        """
        Attempt code generation solution

        Returns:
            (solution, info, error_message)
        """
        try:
            solution, info = self.codegen_solver.solve(
                pde_str, g_init, S, param_values, num_steps
            )

            if solution is None:
                return None, info, info.get('error', 'Code generation failed')

            return solution, info, None

        except Exception as e:
            return None, None, f"Code generation error: {str(e)}"

    def _try_legacy_solve(self, pde_str: str, g_init: np.ndarray, S: np.ndarray,
                         param_values: Dict[str, float], num_steps: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Attempt legacy hardcoded solver (from original pde_solver.py)

        Returns:
            (solution, error_message)
        """
        try:
            from .pde_solver import PDESolver, PDEConfig

            config = PDEConfig(
                dx=self.dx,
                dy=self.dy,
                dt=self.dt,
                boundary_condition=self.boundary_condition
            )
            legacy_solver = PDESolver(config)

            solution, info = legacy_solver.evaluate_pde(
                pde_str, g_init, S, param_values, num_steps
            )

            return solution, None

        except NotImplementedError as e:
            return None, f"Legacy solver doesn't support this form: {str(e)}"
        except Exception as e:
            return None, f"Legacy solver error: {str(e)}"

    def solve(self, pde_str: str, g_init: np.ndarray, S: np.ndarray,
              param_values: Dict[str, float], num_steps: int,
              force_tier: Optional[SolverTier] = None) -> HybridSolverResult:
        """
        Solve PDE using hybrid routing

        Args:
            pde_str: PDE equation string
            g_init: Initial condition (H, W)
            S: Signal field (H, W) or (H, W, T)
            param_values: Parameter values dict
            num_steps: Number of time steps
            force_tier: Force specific solver tier (for testing)

        Returns:
            HybridSolverResult with solution and metadata
        """
        start_time = time.time()
        attempted_tiers = []

        # Routing logic
        if force_tier == SolverTier.SYMBOLIC or (force_tier is None and self.prefer_symbolic):
            if self.verbose:
                print(f"[HybridSolver] Attempting Tier 1: Symbolic solver")

            attempted_tiers.append(SolverTier.SYMBOLIC)
            solution, parse_info, error = self._try_symbolic_solve(
                pde_str, g_init, S, param_values, num_steps
            )

            if solution is not None:
                self.stats['symbolic_success'] += 1
                execution_time = time.time() - start_time

                if self.verbose:
                    print(f"[HybridSolver] ✓ Symbolic solver succeeded ({execution_time:.3f}s)")

                return HybridSolverResult(
                    solution=solution,
                    success=True,
                    tier_used=SolverTier.SYMBOLIC,
                    parse_info=parse_info,
                    execution_time=execution_time,
                    error_message=None,
                    info={'method': 'symbolic', 'parameters': param_values}
                )
            else:
                self.stats['symbolic_failures'] += 1
                if self.verbose:
                    print(f"[HybridSolver] ✗ Symbolic solver failed: {error}")

        # Try code generation if enabled
        if force_tier == SolverTier.CODEGEN or (force_tier is None and self.enable_codegen):
            if self.verbose:
                print(f"[HybridSolver] Attempting Tier 2: Code generation")

            attempted_tiers.append(SolverTier.CODEGEN)
            solution, info, error = self._try_codegen_solve(
                pde_str, g_init, S, param_values, num_steps
            )

            if solution is not None:
                self.stats['codegen_success'] += 1
                execution_time = time.time() - start_time

                if self.verbose:
                    print(f"[HybridSolver] ✓ Code generation succeeded ({execution_time:.3f}s)")

                return HybridSolverResult(
                    solution=solution,
                    success=True,
                    tier_used=SolverTier.CODEGEN,
                    parse_info=None,
                    execution_time=execution_time,
                    error_message=None,
                    info=info
                )
            else:
                self.stats['codegen_failures'] += 1
                if self.verbose:
                    print(f"[HybridSolver] ✗ Code generation failed: {error}")

        # Last resort: try legacy solver
        if force_tier == SolverTier.LEGACY or force_tier is None:
            if self.verbose:
                print(f"[HybridSolver] Attempting Tier 3: Legacy solver")

            attempted_tiers.append(SolverTier.LEGACY)
            solution, error = self._try_legacy_solve(
                pde_str, g_init, S, param_values, num_steps
            )

            if solution is not None:
                self.stats['legacy_success'] += 1
                execution_time = time.time() - start_time

                if self.verbose:
                    print(f"[HybridSolver] ✓ Legacy solver succeeded ({execution_time:.3f}s)")

                return HybridSolverResult(
                    solution=solution,
                    success=True,
                    tier_used=SolverTier.LEGACY,
                    parse_info=None,
                    execution_time=execution_time,
                    error_message=None,
                    info={'method': 'legacy', 'parameters': param_values}
                )
            else:
                if self.verbose:
                    print(f"[HybridSolver] ✗ Legacy solver failed: {error}")

        # All tiers failed
        execution_time = time.time() - start_time
        error_summary = f"All solver tiers failed. Attempted: {[t.value for t in attempted_tiers]}"

        if self.verbose:
            print(f"[HybridSolver] ✗ All tiers failed ({execution_time:.3f}s)")

        return HybridSolverResult(
            solution=None,
            success=False,
            tier_used=attempted_tiers[-1] if attempted_tiers else SolverTier.SYMBOLIC,
            parse_info=None,
            execution_time=execution_time,
            error_message=error_summary,
            info=None
        )

    def get_statistics(self) -> Dict:
        """Get solver usage statistics"""
        total_attempts = sum(self.stats.values())
        return {
            **self.stats,
            'total_attempts': total_attempts,
            'symbolic_rate': self.stats['symbolic_success'] / max(1, total_attempts),
            'codegen_rate': self.stats['codegen_success'] / max(1, total_attempts),
            'legacy_rate': self.stats['legacy_success'] / max(1, total_attempts),
        }

    def fit_parameters(self, pde_str: str, g_init: np.ndarray, S: np.ndarray,
                      g_observed: np.ndarray, param_bounds: Dict[str, Tuple[float, float]],
                      method: str = 'L-BFGS-B') -> Tuple[Dict[str, float], float]:
        """
        Fit PDE parameters to observed data using hybrid solver

        Args:
            pde_str: PDE equation string
            g_init: Initial condition
            S: Signal field
            g_observed: Observed evolution (H, W, T)
            param_bounds: Parameter bounds dict
            method: Optimization method

        Returns:
            (best_params, best_loss)
        """
        from scipy import optimize

        param_names = list(param_bounds.keys())
        bounds = [param_bounds[p] for p in param_names]
        x0 = [(b[0] + b[1]) / 2 for b in bounds]
        num_steps = g_observed.shape[2]

        def objective(params):
            """Objective function for optimization"""
            param_dict = {name: val for name, val in zip(param_names, params)}

            try:
                result = self.solve(pde_str, g_init, S, param_dict, num_steps)

                if not result.success or result.solution is None:
                    return 1e10  # Large penalty

                # Compute MSE
                predicted = result.solution
                if predicted.shape != g_observed.shape:
                    min_T = min(predicted.shape[2], g_observed.shape[2])
                    predicted = predicted[:, :, :min_T]
                    observed = g_observed[:, :, :min_T]
                else:
                    observed = g_observed

                mse = np.mean((predicted - observed) ** 2)
                return mse

            except Exception:
                return 1e10

        # Optimize
        result = optimize.minimize(objective, x0, method=method, bounds=bounds)

        best_params = {name: val for name, val in zip(param_names, result.x)}
        best_loss = result.fun

        return best_params, best_loss
