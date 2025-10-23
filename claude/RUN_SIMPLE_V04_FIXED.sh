#!/bin/bash
# PDE Discovery - AutoGen v0.4 with Context Management
# Fixes context overflow by resetting agent every 50 iterations

echo "=============================================================="
echo "PDE DISCOVERY - AUTOGEN V0.4 (CONTEXT MANAGED)"
echo "=============================================================="
echo ""
echo "✓ Fixes context overflow (resets agent every 50 iterations)"
echo "✓ Works with current vLLM (no restart needed)"
echo "✓ AutoGen v0.4 AssistantAgent"
echo "✓ TensorBoard + Experience Buffer"
echo ""
echo "Ground Truth: ∂g/∂t = α·Δg - β·∇·(g∇(ln S)) + γ·g(1-g/K)"
echo "True Params: α=0.5, β=1.5, γ=0.15, K=3.0"
echo ""
echo "Starting discovery..."
echo "=============================================================="
echo ""

/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_simple_v04_fixed.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --api_base http://localhost:10005/v1 \
  --api_model /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --max_iterations 8000 \
  --samples_per_prompt 4 \
  --reset_interval 50 \
  --output_dir logs/pde_discovery_simple_v04_8k

echo ""
echo "=============================================================="
echo "COMPLETE!"
echo "=============================================================="
echo ""
echo "Results: cat logs/pde_discovery_simple_v04_8k/discovery_results.json"
echo "TensorBoard: tensorboard --logdir logs/pde_discovery_simple_v04_8k/tensorboard --port 6006"
echo ""
