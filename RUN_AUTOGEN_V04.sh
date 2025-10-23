#!/bin/bash
# PDE Discovery - AutoGen v0.4 Implementation
# Execute this to start the full 8000-iteration discovery

echo "=============================================================="
echo "PDE DISCOVERY - AUTOGEN V0.4 - 8000 ITERATIONS"
echo "=============================================================="
echo ""
echo "AutoGen Version: v0.4 (autogen-agentchat)"
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
echo "  ✓ AutoGen v0.4 asynchronous architecture"
echo "  ✓ AssistantAgent with tool support"
echo "  ✓ evaluate_pde tool for PDE evaluation"
echo "  ✓ TensorBoard logging"
echo "  ✓ Experience buffer with top-5 context"
echo "  ✓ Automatic visualization every 200 iterations"
echo ""
echo "API Pattern:"
echo "  • AssistantAgent with tools=[evaluate_pde_tool]"
echo "  • Tools executed directly within agent"
echo "  • No separate executor agent needed"
echo "  • Async/await pattern throughout"
echo ""
echo "Estimated time: 8-12 hours"
echo "=============================================================="
echo ""

# Check if AutoGen v0.4 is installed
echo "Checking AutoGen installation..."
/home/gaoch/miniconda3/envs/llmsr/bin/python -c "
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    print('✓ AutoGen v0.4 is installed correctly')
except ImportError as e:
    print('✗ AutoGen v0.4 not found!')
    print('  Install with: pip install autogen-agentchat autogen-ext[openai]')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "Please install AutoGen v0.4 first:"
    echo "  pip install autogen-agentchat autogen-ext[openai]"
    exit 1
fi

echo ""
echo "Starting discovery..."
echo ""

# Run the discovery
/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_autogen_v04.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --api_base http://localhost:10005/v1 \
  --api_model /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --max_iterations 8000 \
  --samples_per_prompt 4 \
  --output_dir logs/pde_discovery_autogen_v04_8k

echo ""
echo "=============================================================="
echo "DISCOVERY COMPLETE!"
echo "=============================================================="
echo ""
echo "View results:"
echo "  cat logs/pde_discovery_autogen_v04_8k/discovery_results.json"
echo ""
echo "View experience buffer (all PDEs tried):"
echo "  cat logs/pde_discovery_autogen_v04_8k/experience_buffer.json"
echo ""
echo "View with TensorBoard:"
echo "  /home/gaoch/miniconda3/envs/llmsr/bin/tensorboard --logdir logs/pde_discovery_autogen_v04_8k/tensorboard --port 6006"
echo ""
echo "Open browser: http://localhost:6006"
echo ""
