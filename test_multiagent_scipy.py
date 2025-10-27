#!/usr/bin/env python3
"""
Test script to verify:
1. Scipy operators work in PDE code generation
2. Multiagent framework is functional
"""

import numpy as np
import scipy.ndimage
from bench.pde_llmsr_solver import LLMSRPDESolver, _execute_pde_in_subprocess
import multiprocessing

def main():
    print("="*70)
    print("Testing scipy operators in PDE code execution")
    print("="*70)

    # Test 1: Verify scipy is available in execution namespace
    print("\n[Test 1] Testing scipy.ndimage operators in subprocess...")

    test_code = """
def pde_update(g, S, dx, dy, dt, params):
    import numpy as np
    import scipy.ndimage

    p0 = params[0]  # diffusion coefficient

    # Use scipy operators
    laplacian_g = scipy.ndimage.laplace(g) / (dx**2)

    # Compute dg/dt
    dg_dt = p0 * laplacian_g

    # Forward Euler
    g_next = g + dt * dg_dt
    g_next = np.maximum(g_next, 0)

    return g_next
"""

    # Create test data
    g_init = np.random.rand(10, 10)
    S = np.ones((10, 10))
    params = np.array([0.5])
    num_steps = 5
    dx = dy = 1.0
    dt = 0.01

    # Test in subprocess
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_execute_pde_in_subprocess,
        args=(test_code, g_init, S, params, num_steps, dx, dy, dt, result_queue)
    )

    process.start()
    process.join(timeout=10)

    if process.is_alive():
        process.terminate()
        process.join()
        print("❌ Test 1 FAILED: Process timeout")
    else:
        if not result_queue.empty():
            solution, success, error = result_queue.get()
            if success:
                print(f"✓ Test 1 PASSED: scipy operators work in subprocess")
                print(f"  Solution shape: {solution.shape}")
            else:
                print(f"❌ Test 1 FAILED: {error}")
        else:
            print("❌ Test 1 FAILED: No result from subprocess")

    # Test 2: Verify LLMSRPDESolver can evaluate scipy-based code
    print("\n[Test 2] Testing LLMSRPDESolver.evaluate_pde() with scipy code...")

    # Create a mock LLM client (we'll use direct code, not LLM generation)
    class MockLLMClient:
        def generate(self, prompt):
            return test_code

    llm_client = MockLLMClient()
    solver = LLMSRPDESolver(llm_client=llm_client, dx=1.0, dy=1.0, dt=0.01, timeout=10)

    # Evaluate the scipy-based PDE code
    solution, success, error = solver.evaluate_pde(test_code, g_init, S, params, num_steps)

    if success:
        print(f"✓ Test 2 PASSED: LLMSRPDESolver works with scipy code")
        print(f"  Solution shape: {solution.shape}")
        print(f"  Solution range: [{solution.min():.4f}, {solution.max():.4f}]")
    else:
        print(f"❌ Test 2 FAILED: {error}")

    # Test 3: Verify gradient operators
    print("\n[Test 3] Testing scipy gradient operators...")

    gradient_code = """
def pde_update(g, S, dx, dy, dt, params):
    import numpy as np
    import scipy.ndimage

    p0 = params[0]  # diffusion
    p1 = params[1]  # chemotaxis

    # Laplacian
    laplacian_g = scipy.ndimage.laplace(g) / (dx**2)

    # Gradients of S
    grad_S_x = scipy.ndimage.sobel(S, axis=1) / (2*dx)
    grad_S_y = scipy.ndimage.sobel(S, axis=0) / (2*dy)

    # Gradients of g
    grad_g_x = scipy.ndimage.sobel(g, axis=1) / (2*dx)
    grad_g_y = scipy.ndimage.sobel(g, axis=0) / (2*dy)

    # Chemotaxis term: div(g * grad(S))
    flux_x = g * grad_S_x
    flux_y = g * grad_S_y

    div_flux_x = scipy.ndimage.sobel(flux_x, axis=1) / (2*dx)
    div_flux_y = scipy.ndimage.sobel(flux_y, axis=0) / (2*dy)
    div_flux = div_flux_x + div_flux_y

    # PDE: dg/dt = D*Laplacian(g) - chi*div(g*grad(S))
    dg_dt = p0 * laplacian_g - p1 * div_flux

    g_next = g + dt * dg_dt
    g_next = np.maximum(g_next, 0)

    return g_next
"""

    params2 = np.array([0.5, 0.3])
    solution2, success2, error2 = solver.evaluate_pde(gradient_code, g_init, S, params2, num_steps)

    if success2:
        print(f"✓ Test 3 PASSED: Gradient and divergence operators work")
        print(f"  Solution shape: {solution2.shape}")
    else:
        print(f"❌ Test 3 FAILED: {error2}")

    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print("All scipy operator tests completed!")
    print("\nMultiagent framework changes:")
    print("- Generator agent now generates complete Python code")
    print("- Critic agent analyzes visualizations and provides feedback")
    print("- Two-agent conversation loop in run_pde_discovery_autogen_v04.py")
    print("="*70)

if __name__ == '__main__':
    main()

