#!/bin/bash
# PDE Discovery - AutoGen v0.4 WITHOUT Tool Calling + Sliding Window Context
# Works with current vLLM (no restart needed) + Better than agent reset

echo "=============================================================================="
echo "PDE DISCOVERY - AUTOGEN V0.4 (NO TOOL CALLING, SLIDING WINDOW)"
echo "=============================================================================="
echo ""
echo "✓ Works with current vLLM (no restart needed)"
echo "✓ No tool calling required"
echo "✓ Sliding window context (vs. reset loses all learning)"
echo ""
echo "Context Management (Enhanced):"
echo "  • Keeps last 12 messages (conservative to prevent overflow)"
echo "  • Trims every 5 iterations (aggressive)"
echo "  • Truncates long messages to 1000 chars (~200 tokens)"
echo "  • Total: ~12 msgs × 200 tokens = ~2,400 tokens for history"
echo "  • System message + current prompt: ~2,000 tokens"
echo "  • Total request: ~4,400 tokens << 40,730 limit ✅"
echo "  • Better than reset: maintains agent learning!"
echo ""
echo "Ground Truth: ∂g/∂t = α·Δg - β·∇·(g∇(ln S)) + γ·g(1-g/K)"
echo "True Params: α=0.5, β=1.5, γ=0.15, K=3.0"
echo ""
echo "Features:"
echo "  ✓ AutoGen v0.4 AssistantAgent"
echo "  ✓ Text-based equation generation (parsed with regex)"
echo "  ✓ Sliding window conversation history"
echo "  ✓ TensorBoard logging"
echo "  ✓ Experience buffer with top-3 context"
echo "  ✓ Automatic visualization"
echo ""
echo "Estimated time: 8-12 hours"
echo "=============================================================================="
echo ""

/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_simple_v04_sliding.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --api_base http://localhost:10005/v1 \
  --api_model /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --max_iterations 8000 \
  --samples_per_prompt 4 \
  --context_window_size 12 \
  --context_trim_interval 5 \
  --output_dir logs/pde_discovery_simple_v04_sliding_8k

echo ""
echo "=============================================================================="
echo "COMPLETE!"
echo "=============================================================================="
echo ""
echo "Results: cat logs/pde_discovery_simple_v04_sliding_8k/discovery_results.json"
echo "TensorBoard: tensorboard --logdir logs/pde_discovery_simple_v04_sliding_8k/tensorboard --port 6006"
echo ""
echo "Context Stats:"
echo "  - Window size: 20 messages"
echo "  - Trim interval: 10 iterations"
echo "  - Check final history size in results.json"
echo ""
