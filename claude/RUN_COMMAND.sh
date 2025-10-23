#!/bin/bash
# PDE Discovery - Final Running Command
# Execute this to start the full 8000-iteration discovery

echo "=============================================================="
echo "PDE DISCOVERY - STARTING 8000 ITERATION RUN"
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
echo "  ✓ AutoGen dual-agent system"
echo "  ✓ TensorBoard logging"
echo "  ✓ Experience buffer with top-5 context"
echo "  ✓ Automatic visualization"
echo ""
echo "Estimated time: 8-12 hours"
echo "=============================================================="
echo ""

# Run the discovery
/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_final.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --api_base http://localhost:10005/v1 \
  --api_model /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --max_iterations 8000 \
  --samples_per_prompt 4 \
  --use_autogen \
  --output_dir logs/pde_discovery_8k_final

echo ""
echo "=============================================================="
echo "DISCOVERY COMPLETE!"
echo "=============================================================="
echo ""
echo "View results:"
echo "  cat logs/pde_discovery_8k_final/discovery_results.json"
echo ""
echo "View with TensorBoard:"
echo "  /home/gaoch/miniconda3/envs/llmsr/bin/tensorboard --logdir logs/pde_discovery_8k_final/tensorboard --port 6006"
echo ""
