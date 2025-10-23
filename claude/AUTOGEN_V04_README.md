# PDE Discovery with AutoGen v0.4

## ✅ Implementation Complete - AutoGen v0.4

This implementation uses **AutoGen v0.4** (`autogen-agentchat`) with the new asynchronous, event-driven architecture.

---

## 🔄 Key Differences: v0.2 vs v0.4

| Feature | v0.2 (pyautogen) | v0.4 (autogen-agentchat) |
|---------|------------------|--------------------------|
| **Package** | `pyautogen` | `autogen-agentchat` + `autogen-ext` |
| **Import** | `from autogen import ConversableAgent` | `from autogen_agentchat.agents import AssistantAgent` |
| **Model Client** | `llm_config` dict | `OpenAIChatCompletionClient` object |
| **Tool Use** | Separate caller/executor agents | Tools passed directly to AssistantAgent |
| **API Style** | Synchronous | Asynchronous (async/await) |
| **Messages** | `initiate_chat()` | `await agent.on_messages()` |
| **Architecture** | Blocking | Event-driven |

---

## 📦 Installation

```bash
# Install AutoGen v0.4
pip install autogen-agentchat autogen-ext[openai]

# Install other dependencies
pip install tensorboard torch torchvision pillow
```

**Verify installation:**
```bash
python -c "from autogen_agentchat.agents import AssistantAgent; print('✓ AutoGen v0.4 installed')"
```

---

## 🎯 How It Works (v0.4)

### 1. Model Client Setup

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient

# For vLLM (OpenAI-compatible API)
model_client = OpenAIChatCompletionClient(
    model="/mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct",
    base_url="http://localhost:10005/v1",
    api_key="EMPTY",  # vLLM doesn't need real key
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
    },
)
```

### 2. Tool Definition

```python
def evaluate_pde_tool(
    equation: Annotated[str, "PDE equation to evaluate"]
) -> str:
    """Evaluate PDE by fitting parameters and computing metrics"""

    # 1. Fit parameters
    fitted_params, loss = solver.fit_pde_parameters(...)

    # 2. Evaluate PDE
    predicted, info = solver.evaluate_pde(...)

    # 3. Compute metrics
    r2 = compute_spatiotemporal_loss(...)
    score = r2 * 10 * (1 - mass_error)

    # 4. Return JSON result
    return json.dumps({
        'success': True,
        'score': score,
        'r2': r2,
        'message': f"Score: {score:.4f}"
    })
```

### 3. AssistantAgent with Tools

```python
from autogen_agentchat.agents import AssistantAgent

assistant = AssistantAgent(
    name="PDE_Generator",
    model_client=model_client,
    tools=[evaluate_pde_tool],  # Tools executed directly in agent
    reflect_on_tool_use=True,   # Reflect on tool results
    system_message="""You are an expert in PDE modeling.

Generate PDE candidates and call evaluate_pde(equation="...") for each.
Analyze results and propose improved candidates."""
)
```

### 4. Async Conversation

```python
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

async def discover():
    cancellation_token = CancellationToken()

    # Send message to agent
    response = await assistant.on_messages(
        [TextMessage(content="Generate 4 PDE candidates", source="user")],
        cancellation_token
    )

    print(response.chat_message.content)

    # Close model client
    await model_client.close()

# Run async function
asyncio.run(discover())
```

---

## 🚀 Running the Discovery

### Quick Test (50 iterations, ~5 minutes)

```bash
/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_autogen_v04.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --max_iterations 50 \
  --samples_per_prompt 2 \
  --output_dir logs/pde_test_v04
```

### Full Run (8000 iterations, 8-12 hours)

```bash
./RUN_AUTOGEN_V04.sh
```

Or directly:
```bash
/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_autogen_v04.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --api_base http://localhost:10005/v1 \
  --api_model /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --max_iterations 8000 \
  --samples_per_prompt 4 \
  --output_dir logs/pde_discovery_autogen_v04_8k
```

### Background Run

```bash
nohup ./RUN_AUTOGEN_V04.sh > logs/autogen_v04_discovery.log 2>&1 &

# Monitor
tail -f logs/autogen_v04_discovery.log
```

---

## 📊 Monitoring with TensorBoard

```bash
/home/gaoch/miniconda3/envs/llmsr/bin/tensorboard \
  --logdir logs/pde_discovery_autogen_v04_8k/tensorboard \
  --port 6006
```

Open: `http://localhost:6006`

**Metrics:**
- `metrics/score` - Overall quality (0-10)
- `metrics/r2` - R² coefficient
- `metrics/mse` - Mean squared error
- `metrics/mass_error` - Mass conservation error (%)
- `best/score` - Best score so far
- `best/r2` - Best R² so far
- `visualizations/best` - Best PDE visualizations
- `performance/*` - Timing and buffer stats

---

## 🎓 Ground Truth Challenge

**Target PDE:**
```
∂g/∂t = α·Δg - β·∇·(g∇(ln S)) + γ·g(1-g/K)
```

**True Parameters:**
- α = 0.5   (diffusion)
- β = 1.5   (chemotaxis)
- γ = 0.15  (growth rate)
- K = 3.0   (carrying capacity)

**Test Data:** `logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5`
- 15.1% mass increase
- 1.13x peak growth
- Dynamic chemotaxis + diffusion + logistic growth

---

## 📈 Expected Progress

| Iteration | Expected R² | What's Discovered |
|-----------|-------------|-------------------|
| 100 | 0.70 | Basic diffusion: α·Δg |
| 500 | 0.85 | + Chemotaxis: -β·∇·(g∇S) |
| 2000 | 0.92 | + Growth: +γ·g |
| 5000 | 0.95+ | Full PDE with 4 parameters |
| 8000 | 0.97-0.98 | Near-perfect match |

---

## 📁 Output Structure

```
logs/pde_discovery_autogen_v04_8k/
├── tensorboard/              # TensorBoard logs
├── discovery_results.json    # Final results
├── experience_buffer.json    # All PDEs tried
├── best_iter_000200.png     # Visualization @ iter 200
├── best_iter_000400.png     # Visualization @ iter 400
└── ...
```

---

## 🔍 Checking Results

### View Final Results
```bash
cat logs/pde_discovery_autogen_v04_8k/discovery_results.json
```

### View Top-5 PDEs
```bash
/home/gaoch/miniconda3/envs/llmsr/bin/python -c "
import json

with open('logs/pde_discovery_autogen_v04_8k/experience_buffer.json') as f:
    data = json.load(f)

exps = sorted(data['experiences'], key=lambda x: x['score'], reverse=True)

print('TOP 5 DISCOVERED PDEs:')
print('='*70)
for i, exp in enumerate(exps[:5], 1):
    print(f'{i}. Score: {exp[\"score\"]:.4f} | R²: {exp[\"metrics\"][\"r2\"]:.4f}')
    print(f'   {exp[\"equation\"]}')
    print(f'   Params: {exp[\"parameters\"]}')
    print()
"
```

---

## 🔧 AutoGen v0.4 Code Patterns

### Pattern 1: Single AssistantAgent with Tools

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

def my_tool(param: str) -> str:
    return f"Result for {param}"

async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key="sk-xxx"
    )

    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[my_tool],
        reflect_on_tool_use=True
    )

    response = await assistant.on_messages(
        [TextMessage(content="Use my_tool with 'test'", source="user")],
        CancellationToken()
    )

    print(response.chat_message.content)
    await model_client.close()

asyncio.run(main())
```

### Pattern 2: Custom vLLM Endpoint

```python
model_client = OpenAIChatCompletionClient(
    model="/path/to/model",
    base_url="http://localhost:10005/v1",
    api_key="EMPTY",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
    }
)
```

### Pattern 3: Tool Returns JSON String

```python
def evaluate_tool(data: str) -> str:
    """Tool must return string (can be JSON)"""
    result = {
        'status': 'success',
        'value': 42,
        'message': 'Computed successfully'
    }
    return json.dumps(result)
```

---

## 🐛 Troubleshooting

### Error: "Cannot import AssistantAgent"

```bash
# Wrong package installed
pip uninstall pyautogen autogen

# Install correct v0.4 packages
pip install autogen-agentchat autogen-ext[openai]
```

### Error: "Model doesn't support function calling"

Make sure your vLLM model supports function calling:
```python
model_info={
    "function_calling": True,  # Set to True
}
```

### Error: "RuntimeError: This event loop is already running"

Use `asyncio.run()` not `await` in main:
```python
# Correct
def main():
    asyncio.run(main_async())

# Wrong
async def main():
    await main_async()  # Already in async context
```

### TensorBoard Not Working

```bash
pip install tensorboard torch torchvision pillow
```

### Too Slow

- Reduce `--max_iterations` to 2000
- Reduce `--samples_per_prompt` to 2
- Use faster model

---

## ✅ Migration Checklist

If migrating from v0.2 code:

- [ ] Change package: `pyautogen` → `autogen-agentchat` + `autogen-ext`
- [ ] Update imports: `autogen` → `autogen_agentchat` / `autogen_ext`
- [ ] Replace `llm_config` dict with `OpenAIChatCompletionClient` object
- [ ] Change `ConversableAgent` → `AssistantAgent`
- [ ] Pass tools directly to agent, remove executor agent
- [ ] Make functions async with `async def` and `await`
- [ ] Replace `initiate_chat()` with `await agent.on_messages()`
- [ ] Use `TextMessage` instead of dict messages
- [ ] Add `CancellationToken()` parameter
- [ ] Call `await model_client.close()` at end
- [ ] Use `asyncio.run()` to start async main

---

## 📚 Resources

- **AutoGen v0.4 Docs**: https://microsoft.github.io/autogen/stable/
- **Migration Guide**: `autogen.md` (in this repo)
- **Tool Use Example**: Lines 539-609 in `autogen.md`
- **AssistantAgent Docs**: https://microsoft.github.io/autogen/stable/reference/python/autogen_agentchat/agents/AssistantAgent.html

---

## 🎉 Ready to Run!

Everything is set up with the correct AutoGen v0.4 implementation.

**Start discovery:**
```bash
./RUN_AUTOGEN_V04.sh
```

**Monitor:**
```bash
/home/gaoch/miniconda3/envs/llmsr/bin/tensorboard \
  --logdir logs/pde_discovery_autogen_v04_8k/tensorboard \
  --port 6006
```

**Check progress:**
```bash
tail -f logs/autogen_v04_discovery.log  # If running in background
```

---

**Good luck discovering the PDE with AutoGen v0.4! 🚀**
