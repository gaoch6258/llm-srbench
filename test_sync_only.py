#!/usr/bin/env python3
"""
Simplest possible test - sync only, no async context
"""
import subprocess
import sys

# Run the real test in a subprocess to avoid async context issues
result = subprocess.run(
    [sys.executable, "-c", """
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, '/home/gaoch/llm-srbench')

print("Testing without async context...")

from run_pde_discovery_autogen_v04 import AsyncLLMClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily
from bench.pde_llmsr_solver import LLMSRPDESolver

# Create model client (this will be used in subprocess)
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

solver = LLMSRPDESolver(llm_client=llm_client, dx=1.0, dy=1.0, dt=0.01, timeout=30)
print("✓ LLMSRPDESolver created")

# Test code generation (this will call LLM)
print("\\nTesting PDE code generation (calling LLM)...")
code, error = solver.generate_pde_code("Simple diffusion with parameter p0", num_params=1)

if error:
    print(f"✗ Code generation failed: {error}")
    sys.exit(1)

print("✓ Code generated successfully!")
print(f"\\nCode length: {len(code)} characters")
print(f"Code preview: {code[:200]}...")

print("\\n✅ Test passed!")
"""],
    capture_output=True,
    text=True,
    timeout=120
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print(f"Exit code: {result.returncode}")
