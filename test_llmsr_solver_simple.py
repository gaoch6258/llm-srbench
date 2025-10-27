#!/usr/bin/env python3
"""
Simple test of LLMSRPDESolver with LLM code generation
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Import AsyncLLMClient from the main script
from run_pde_discovery_autogen_v04 import AsyncLLMClient

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily
import asyncio


async def main():
    print("Testing LLMSRPDESolver...")

    # Create model client
    model_client = OpenAIChatCompletionClient(
        model="/mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct",
        base_url="http://localhost:10005/v1",
        api_key="EMPTY",
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
    )
    print("✓ Model client created")

    # Create AsyncLLMClient
    llm_client = AsyncLLMClient(model_client)
    print("✓ AsyncLLMClient created")

    # Import LLMSRPDESolver
    from bench.pde_llmsr_solver import LLMSRPDESolver

    solver = LLMSRPDESolver(llm_client=llm_client, dx=1.0, dy=1.0, dt=0.01, timeout=60)
    print("✓ LLMSRPDESolver created")

    # Test code generation
    print("\nTesting PDE code generation...")
    code, error = solver.generate_pde_code("Diffusion with coefficient p0", num_params=1)

    if error:
        print(f"✗ Code generation failed: {error}")
        return

    print("✓ Code generated successfully!")
    print("\nGenerated code preview:")
    print("=" * 70)
    print(code[:500])
    print("=" * 70)

    # Test evaluation with simple data
    print("\nTesting PDE evaluation...")
    g_init = np.random.rand(32, 32) * 0.1 + 0.5
    S = np.ones((32, 32))
    params = np.array([0.5])  # diffusion coefficient
    num_steps = 10

    solution, success, error_msg = solver.evaluate_pde(code, g_init, S, params, num_steps)

    if not success:
        print(f"✗ Evaluation failed: {error_msg}")
        return

    print(f"✓ Evaluation successful!")
    print(f"  Solution shape: {solution.shape}")
    print(f"  Min/Max: {solution.min():.4f} / {solution.max():.4f}")

    # Close model client
    await model_client.close()

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
