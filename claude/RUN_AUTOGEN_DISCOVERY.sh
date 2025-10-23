#!/bin/bash
# PDE Discovery - AutoGen Tool Use Version
# Execute this to start the full 8000-iteration discovery with AutoGen tool calling

echo "=============================================================="
echo "PDE DISCOVERY - AUTOGEN TOOL USE - 8000 ITERATIONS"
echo "=============================================================="
echo ""
echo "Ground Truth PDE:"
echo "  ∂g/∂t = α·Δg - β·∇·(g∇(ln S)) + γ·g(1-g/K)"
echo ""
echo "True Parameters:"
echo "  α = 0.5   (diffusion)"
echo "  β = 1.5   (chemotaxis)"
echo "  γ = 0.15  (growth)"
echo "  K = 3.0   (capacity)"
echo ""
echo "Features:"
echo "  ✓ 8000 iterations"
echo "  ✓ AutoGen 0.2 (pyautogen) with tool use"
echo "  ✓ evaluate_pde tool registered with Generator agent"
echo "  ✓ TensorBoard logging"
echo "  ✓ Experience buffer with top-5 context"
echo "  ✓ Automatic visualization every 200 iterations"
echo ""
echo "AutoGen Pattern:"
echo "  • Generator agent (PDE_Generator) proposes candidates"
echo "  • Calls evaluate_pde tool for each candidate"
echo "  • Executor agent (PDE_Executor) runs the tool"
echo "  • Tool fits parameters and computes metrics"
echo "  • Results stored in experience buffer"
echo ""
echo "Estimated time: 8-12 hours"
echo "=============================================================="
echo ""

# Run the discovery
/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_autogen.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --api_base http://localhost:10005/v1 \
  --api_model /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --max_iterations 8000 \
  --samples_per_prompt 4 \
  --output_dir logs/pde_discovery_autogen_8k

echo ""
echo "=============================================================="
echo "DISCOVERY COMPLETE!"
echo "=============================================================="
echo ""
echo "View results:"
echo "  cat logs/pde_discovery_autogen_8k/discovery_results.json"
echo ""
echo "View experience buffer (all PDEs tried):"
echo "  cat logs/pde_discovery_autogen_8k/experience_buffer.json"
echo ""
echo "View with TensorBoard:"
echo "  /home/gaoch/miniconda3/envs/llmsr/bin/tensorboard --logdir logs/pde_discovery_autogen_8k/tensorboard --port 6006"
echo ""
echo "Open browser: http://localhost:6006"
echo ""
