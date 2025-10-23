#!/bin/bash
# PDE Discovery - AutoGen v0.4 WITHOUT Tool Calling
# Works with current vLLM server (no --enable-auto-tool-choice needed)

echo "=============================================================="
echo "PDE DISCOVERY - AUTOGEN V0.4 (NO TOOL CALLING)"
echo "=============================================================="
echo ""
echo "This version works with your current vLLM server!"
echo "No need to restart vLLM with special flags."
echo ""
echo "Ground Truth: ∂g/∂t = α·Δg - β·∇·(g∇(ln S)) + γ·g(1-g/K)"
echo "True Params: α=0.5, β=1.5, γ=0.15, K=3.0"
echo ""
echo "Features:"
echo "  ✓ AutoGen v0.4 AssistantAgent"
echo "  ✓ Text-based equation generation (no tool calling)"
echo "  ✓ Pattern extraction from agent responses"
echo "  ✓ TensorBoard logging"
echo "  ✓ Experience buffer"
echo ""
echo "Starting discovery..."
echo "=============================================================="
echo ""

/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_simple_v04.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --api_base http://localhost:10005/v1 \
  --api_model /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --max_iterations 8000 \
  --samples_per_prompt 4 \
  --output_dir logs/pde_discovery_simple_v04_8k

echo ""
echo "=============================================================="
echo "COMPLETE!"
echo "=============================================================="
echo ""
echo "Results: cat logs/pde_discovery_simple_v04_8k/discovery_results.json"
echo "TensorBoard: tensorboard --logdir logs/pde_discovery_simple_v04_8k/tensorboard --port 6006"
echo ""
