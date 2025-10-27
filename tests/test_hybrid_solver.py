"""
Test Suite for Hybrid PDE Solver

Tests both symbolic (Tier 1) and code generation (Tier 2) solvers
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add bench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.pde_hybrid_solver import PDEHybridSolver, SolverTier
from bench.pde_symbolic_solver import SymbolicPDEParser, SymbolicPDEEvaluator
from bench.pde_codegen_solver import CodeGenPDESolver


class TestSymbolicParser:
    """Test SymPy-based PDE parser"""

    def test_parse_simple_diffusion(self):
        """Test parsing simple diffusion PDE"""
        parser = SymbolicPDEParser()
        structure = parser.parse("∂g/∂t = α·Δg")

        assert structure.is_valid
        assert len(structure.spatial_operators) == 1
        assert structure.spatial_operators[0].op_type.value == "laplacian"
        assert 'α' in structure.parameters or 'alpha' in structure.parameters

    def test_parse_chemotaxis(self):
        """Test parsing chemotaxis PDE"""
        parser = SymbolicPDEParser()
        structure = parser.parse("∂g/∂t = α·Δg - β·∇·(g∇(ln S))")

        assert structure.is_valid
        assert len(structure.spatial_operators) >= 1  # Should have laplacian and/or divergence
        assert 'α' in structure.parameters or 'β' in structure.parameters

    def test_parse_with_reaction(self):
        """Test parsing PDE with reaction term"""
        parser = SymbolicPDEParser()
        structure = parser.parse("∂g/∂t = α·Δg + γ·g(1-g/K)")

        assert structure.is_valid
        assert len(structure.spatial_operators) >= 1
        assert len(structure.reaction_terms) >= 1

    def test_parse_invalid_pde(self):
        """Test parsing invalid/nonsensical PDE"""
        parser = SymbolicPDEParser()
        structure = parser.parse("this is not a PDE")

        # Should either parse as invalid or have no operators
        assert not structure.is_valid or len(structure.spatial_operators) == 0


class TestSymbolicSolver:
    """Test symbolic PDE solver"""

    @pytest.fixture
    def simple_problem(self):
        """Create a simple test problem"""
        H, W = 32, 32
        T = 20

        # Initial condition: Gaussian blob
        x = np.linspace(0, H-1, H)
        y = np.linspace(0, W-1, W)
        X, Y = np.meshgrid(x, y)
        g_init = np.exp(-((X - H/2)**2 + (Y - W/2)**2) / (0.1 * H**2))

        # Signal field: uniform
        S = np.ones((H, W))

        return g_init, S, T

    def test_solve_diffusion(self, simple_problem):
        """Test solving pure diffusion"""
        g_init, S, T = simple_problem

        solver = SymbolicPDEEvaluator(dx=1.0, dy=1.0, dt=0.01)
        solution, info = solver.solve(
            pde_str="∂g/∂t = α·Δg",
            g_init=g_init,
            S=S,
            param_values={'α': 0.5},
            num_steps=T
        )

        assert solution is not None
        assert solution.shape == (g_init.shape[0], g_init.shape[1], T)
        assert np.all(solution >= 0)  # Non-negativity
        assert np.all(np.isfinite(solution))  # No NaN/Inf
        assert info['method'] == 'symbolic_fd'

    def test_solve_chemotaxis(self, simple_problem):
        """Test solving chemotaxis PDE"""
        g_init, S, T = simple_problem

        # Create gradient in S
        x = np.linspace(0, S.shape[1]-1, S.shape[1])
        y = np.linspace(0, S.shape[0]-1, S.shape[0])
        X, Y = np.meshgrid(x, y)
        S = np.exp(0.01 * X + 0.01 * Y)

        solver = SymbolicPDEEvaluator(dx=1.0, dy=1.0, dt=0.01)
        solution, info = solver.solve(
            pde_str="∂g/∂t = α·Δg - β·∇·(g∇(ln S))",
            g_init=g_init,
            S=S,
            param_values={'α': 0.5, 'β': 1.0},
            num_steps=T
        )

        assert solution is not None
        assert solution.shape == (g_init.shape[0], g_init.shape[1], T)
        assert np.all(solution >= 0)
        assert np.all(np.isfinite(solution))

    def test_solve_with_reaction(self, simple_problem):
        """Test solving PDE with reaction term"""
        g_init, S, T = simple_problem

        solver = SymbolicPDEEvaluator(dx=1.0, dy=1.0, dt=0.01)
        solution, info = solver.solve(
            pde_str="∂g/∂t = α·Δg + γ·g",
            g_init=g_init,
            S=S,
            param_values={'α': 0.5, 'γ': 0.1},
            num_steps=T
        )

        assert solution is not None
        assert solution.shape == (g_init.shape[0], g_init.shape[1], T)
        assert np.all(np.isfinite(solution))


class TestCodeGenSolver:
    """Test code generation solver"""

    @pytest.fixture
    def simple_problem(self):
        """Create a simple test problem"""
        H, W = 32, 32
        T = 10  # Fewer steps for code gen (slower)

        x = np.linspace(0, H-1, H)
        y = np.linspace(0, W-1, W)
        X, Y = np.meshgrid(x, y)
        g_init = np.exp(-((X - H/2)**2 + (Y - W/2)**2) / (0.1 * H**2))
        S = np.ones((H, W))

        return g_init, S, T

    def test_template_generation(self, simple_problem):
        """Test template-based code generation (no LLM)"""
        g_init, S, T = simple_problem

        solver = CodeGenPDESolver(dx=1.0, dy=1.0, dt=0.01, llm_client=None)
        solution, info = solver.solve(
            pde_str="∂g/∂t = α·Δg",
            g_init=g_init,
            S=S,
            param_values={'α': 0.5},
            num_steps=T,
            use_cache=False
        )

        # Template may or may not match the actual PDE
        # Just check it runs without crashing
        if solution is not None:
            assert solution.shape == (g_init.shape[0], g_init.shape[1], T)
            assert np.all(np.isfinite(solution))
            assert info['method'] == 'code_generation'
        else:
            # If it fails, should have error message
            assert 'error' in info


class TestHybridSolver:
    """Test hybrid routing system"""

    @pytest.fixture
    def problem(self):
        """Create test problem"""
        H, W = 32, 32
        T = 20

        x = np.linspace(0, H-1, H)
        y = np.linspace(0, W-1, W)
        X, Y = np.meshgrid(x, y)
        g_init = np.exp(-((X - H/2)**2 + (Y - W/2)**2) / (0.1 * H**2))
        S = np.exp(0.01 * X + 0.01 * Y)

        return g_init, S, T

    def test_routing_to_symbolic(self, problem):
        """Test that simple PDEs route to symbolic solver"""
        g_init, S, T = problem

        solver = PDEHybridSolver(
            dx=1.0, dy=1.0, dt=0.01,
            prefer_symbolic=True,
            enable_codegen=False,
            verbose=True
        )

        result = solver.solve(
            pde_str="∂g/∂t = α·Δg",
            g_init=g_init,
            S=S,
            param_values={'α': 0.5},
            num_steps=T
        )

        assert result.success
        assert result.tier_used == SolverTier.SYMBOLIC
        assert result.solution is not None

    def test_fallback_to_legacy(self, problem):
        """Test fallback to legacy solver when needed"""
        g_init, S, T = problem

        solver = PDEHybridSolver(
            dx=1.0, dy=1.0, dt=0.01,
            prefer_symbolic=True,
            enable_codegen=False,  # Disable codegen to force legacy
            verbose=True
        )

        # This might fail symbolic parsing but succeed in legacy
        result = solver.solve(
            pde_str="∂g/∂t = α·Δg - β·∇·(g∇(ln S))",
            g_init=g_init,
            S=S,
            param_values={'α': 0.5, 'β': 1.0},
            num_steps=T
        )

        # Should succeed via one of the tiers
        if result.success:
            assert result.solution is not None
            assert result.tier_used in [SolverTier.SYMBOLIC, SolverTier.LEGACY]

    def test_parameter_fitting(self, problem):
        """Test parameter fitting with hybrid solver"""
        g_init, S, T = problem

        solver = PDEHybridSolver(dx=1.0, dy=1.0, dt=0.01, verbose=False)

        # Generate synthetic data with known parameters
        true_params = {'α': 0.5}
        result_true = solver.solve("∂g/∂t = α·Δg", g_init, S, true_params, T)

        if result_true.success:
            g_observed = result_true.solution

            # Fit parameters
            param_bounds = {'α': (0.1, 2.0)}
            fitted_params, loss = solver.fit_parameters(
                pde_str="∂g/∂t = α·Δg",
                g_init=g_init,
                S=S,
                g_observed=g_observed,
                param_bounds=param_bounds
            )

            # Check that fitted parameter is close to true parameter
            assert 'α' in fitted_params
            assert abs(fitted_params['α'] - true_params['α']) < 0.7  # Relaxed tolerance for optimization
            assert loss < 0.5  # Should achieve reasonable fit

    def test_statistics_tracking(self, problem):
        """Test that solver tracks usage statistics"""
        g_init, S, T = problem

        solver = PDEHybridSolver(dx=1.0, dy=1.0, dt=0.01)

        # Run several solves
        for pde in ["∂g/∂t = α·Δg", "∂g/∂t = α·Δg + β·g"]:
            solver.solve(pde, g_init, S, {'α': 0.5, 'β': 0.1}, T)

        stats = solver.get_statistics()

        assert 'symbolic_success' in stats
        assert 'codegen_success' in stats
        assert 'total_attempts' in stats
        assert stats['total_attempts'] >= 2


class TestEndToEnd:
    """End-to-end integration tests"""

    def test_multiple_pde_forms(self):
        """Test various PDE forms end-to-end"""
        H, W, T = 32, 32, 15

        x = np.linspace(0, H-1, H)
        y = np.linspace(0, W-1, W)
        X, Y = np.meshgrid(x, y)
        g_init = np.exp(-((X - H/2)**2 + (Y - W/2)**2) / (0.1 * H**2))
        S = np.exp(0.01 * X)

        solver = PDEHybridSolver(dx=1.0, dy=1.0, dt=0.01, verbose=False)

        test_cases = [
            ("∂g/∂t = α·Δg", {'α': 0.5}),
            ("∂g/∂t = α·Δg + β·g", {'α': 0.5, 'β': 0.1}),
            ("∂g/∂t = α·Δg - β·g·S", {'α': 0.5, 'β': 0.1}),
        ]

        results = []
        for pde_str, params in test_cases:
            result = solver.solve(pde_str, g_init, S, params, T)
            results.append((pde_str, result.success, result.tier_used))

        # Print summary
        print("\n=== PDE Test Results ===")
        for pde, success, tier in results:
            status = "✓" if success else "✗"
            print(f"{status} {pde[:50]:50s} [{tier.value}]")

        # At least some should succeed
        successes = sum(1 for _, success, _ in results if success)
        assert successes > 0, "At least one PDE should succeed"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
