#!/bin/bash
# PDE Discovery - AutoGen v0.4 with Sliding Window Context Management
# Refined version: Preserves recent conversation instead of resetting agent

echo "=============================================================================="
echo "PDE DISCOVERY - AUTOGEN V0.4 - SLIDING WINDOW CONTEXT"
echo "=============================================================================="
echo ""
echo "AutoGen Version: v0.4 (autogen-agentchat)"
echo ""
echo "Context Management Strategy:"
echo "  ✓ Sliding window: Keeps last 20 messages"
echo "  ✓ Preserves recent learning (vs. reset loses all context)"
echo "  ✓ Prevents context overflow (40,730 token limit)"
echo "  ✓ Trims every 10 iterations"
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
echo "  ✓ Sliding window context (20 messages)"
echo "  ✓ TensorBoard logging"
echo "  ✓ Experience buffer with top-3 context"
echo "  ✓ Automatic visualization every 200 iterations"
echo ""
echo "API Pattern:"
echo "  • AssistantAgent with tools=[evaluate_pde_tool]"
echo "  • Tools executed directly within agent"
echo "  • No separate executor agent needed"
echo "  • Async/await pattern throughout"
echo "  • Conversation history trimmed periodically"
echo ""
echo "Estimated time: 8-12 hours"
echo "=============================================================================="
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

# Check if vLLM supports tool calling
echo ""
echo "Checking vLLM tool calling support..."
echo "NOTE: This requires vLLM server started with:"
echo "  --enable-auto-tool-choice --tool-call-parser hermes"
echo ""
echo "If you get 'auto tool choice requires --enable-auto-tool-choice' error:"
echo "  1. Stop vLLM: pkill -f 'vllm serve'"
echo "  2. Start with: ./engine_with_tools.sh"
echo ""

read -p "Is vLLM running with tool support? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Please start vLLM with tool support first:"
    echo "  pkill -f 'vllm serve'"
    echo "  ./engine_with_tools.sh"
    echo ""
    echo "Or use the simple version (no tool calling):"
    echo "  ./RUN_SIMPLE_V04_FIXED.sh"
    exit 1
fi

echo ""
echo "Starting discovery with sliding window context management..."
echo ""

# Run the discovery
/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_autogen_v04_sliding.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --api_base http://localhost:10005/v1 \
  --api_model /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --max_iterations 8000 \
  --samples_per_prompt 4 \
  --context_window_size 15 \
  --context_trim_interval 15 \
  --output_dir logs/pde_discovery_autogen_v04_sliding_8k

echo ""
echo "=============================================================================="
echo "DISCOVERY COMPLETE!"
echo "=============================================================================="
echo ""
echo "View results:"
echo "  cat logs/pde_discovery_autogen_v04_sliding_8k/discovery_results.json"
echo ""
echo "View experience buffer (all PDEs tried):"
echo "  cat logs/pde_discovery_autogen_v04_sliding_8k/experience_buffer.json"
echo ""
echo "View with TensorBoard:"
echo "  /home/gaoch/miniconda3/envs/llmsr/bin/tensorboard --logdir logs/pde_discovery_autogen_v04_sliding_8k/tensorboard --port 6006"
echo ""
echo "Open browser: http://localhost:6006"
echo ""
echo "Context Management Stats:"
echo "  - Window size: 20 messages"
echo "  - Trim interval: 10 iterations"
echo "  - Final history size: (see results.json)"
echo ""
