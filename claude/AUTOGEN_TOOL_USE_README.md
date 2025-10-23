# PDE Discovery with AutoGen Tool Use

## ✅ Implementation Complete

This implementation uses **AutoGen 0.2 (pyautogen)** with the proper **tool use pattern** as documented in `autogen.md`.

---

## 🎯 Key Features

### 1. **Correct AutoGen API**
- Uses `from autogen import ConversableAgent, register_function`
- Proper `llm_config` with `config_list` format
- Compatible with vLLM custom endpoint

### 2. **Tool Use Pattern**
- `evaluate_pde` function registered as a tool
- **Generator agent** (`PDE_Generator`) proposes PDE candidates and suggests tool calls
- **Executor agent** (`PDE_Executor`) executes the tool
- Tool fits parameters, computes metrics, updates buffer, logs to TensorBoard

### 3. **AutoGen Conversation Flow**
```
Executor → Generator: "Propose PDE candidates for chemotaxis data..."
Generator → evaluate_pde(equation="α·Δg - β·∇·(g∇(ln S))")
Executor → Runs tool → Returns {"score": 8.5, "r2": 0.92, ...}
Generator → Analyzes results → Proposes refined candidate
... (iterates)
```

### 4. **Integration with Existing System**
- ✅ Experience buffer with top-5 context
- ✅ TensorBoard logging (metrics + images)
- ✅ Visualization every 200 iterations
- ✅ Convergence detection
- ✅ Plateau patience mechanism

---

## 📁 Files

### Main Script
**`run_pde_discovery_autogen.py`**
- Full implementation with AutoGen tool use
- 384 lines, well-documented
- Fallback-free (requires pyautogen)

### Shell Script
**`RUN_AUTOGEN_DISCOVERY.sh`**
- Convenient wrapper for 8000 iteration run
- Uses complex test case v2 (15% mass increase)
- Includes TensorBoard instructions

---

## 🚀 Running the Discovery

### Quick Test (50 iterations, ~5 minutes)
```bash
/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_autogen.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --api_base http://localhost:10005/v1 \
  --api_model /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --max_iterations 50 \
  --samples_per_prompt 2 \
  --output_dir logs/pde_test_autogen
```

### Full Run (8000 iterations, 8-12 hours)
```bash
./RUN_AUTOGEN_DISCOVERY.sh
```

Or directly:
```bash
/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_autogen.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --api_base http://localhost:10005/v1 \
  --api_model /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --max_iterations 8000 \
  --samples_per_prompt 4 \
  --output_dir logs/pde_discovery_autogen_8k
```

### Background Run with Logging
```bash
nohup ./RUN_AUTOGEN_DISCOVERY.sh > logs/autogen_discovery.log 2>&1 &

# Monitor progress
tail -f logs/autogen_discovery.log
```

---

## 📊 Monitoring with TensorBoard

Launch TensorBoard while discovery is running:

```bash
/home/gaoch/miniconda3/envs/llmsr/bin/tensorboard \
  --logdir logs/pde_discovery_autogen_8k/tensorboard \
  --port 6006
```

Open browser: `http://localhost:6006`

**Metrics Available:**
- `metrics/score` - Overall PDE quality (0-10)
- `metrics/r2` - R² coefficient
- `metrics/mse` - Mean squared error
- `metrics/mass_error` - Mass conservation error (%)
- `best/score` - Best score so far
- `best/r2` - Best R² so far
- `visualizations/best` - Best PDE visualizations
- `performance/iteration_time` - Time per iteration
- `performance/buffer_size` - Experience buffer growth
- `performance/plateau_counter` - Plateau detection

---

## 🔧 How It Works

### 1. Tool Registration

```python
from autogen import register_function

register_function(
    self.evaluate_pde_tool,
    caller=self.generator_agent,  # Agent that suggests tool calls
    executor=self.executor_agent,  # Agent that executes tool calls
    name="evaluate_pde",
    description="Evaluate a PDE candidate by fitting parameters and computing metrics"
)
```

### 2. Tool Function

```python
def evaluate_pde_tool(
    self,
    equation: Annotated[str, "The PDE equation string to evaluate"]
) -> dict:
    """Evaluate PDE candidate - registered as AutoGen tool"""

    # 1. Fit parameters via optimization
    fitted_params, loss = self.solver.fit_pde_parameters(
        equation, problem.g_init, problem.S, problem.g_observed,
        param_bounds={'α': (0.01, 3.0), 'β': (0.01, 3.0), ...}
    )

    # 2. Evaluate PDE with fitted parameters
    predicted, info = self.solver.evaluate_pde(
        equation, problem.g_init, problem.S, fitted_params,
        num_steps=problem.g_observed.shape[2]
    )

    # 3. Compute metrics
    mse = self.solver.compute_spatiotemporal_loss(predicted, problem.g_observed, 'mse')
    r2 = self.solver.compute_spatiotemporal_loss(predicted, problem.g_observed, 'r2')

    # 4. Compute score
    score = r2 * 10 * (1 - min(mass_error / 100, 0.5))

    # 5. Update buffer and TensorBoard
    self.buffer.add(equation, score, metrics, ...)
    self.writer.add_scalar('metrics/score', score, self.iteration)

    # 6. Return results to agent
    return {
        'success': True,
        'score': score,
        'r2': r2,
        'message': f"Score: {score:.4f}, R²: {r2:.4f}"
    }
```

### 3. Conversation Loop

```python
for iteration in range(1, max_iterations + 1):
    # Get top-5 context from experience buffer
    experience_context = self.buffer.format_for_prompt(k=5)

    # Create prompt with data summary + context
    prompt = f"""Discover PDE for chemotaxis from {shape} data.

Data Summary:
- Cell density range: [{g_min}, {g_max}]
- Mass change: {mass_change_pct}%

{experience_context}

Generate {samples_per_prompt} novel PDE candidates and evaluate them
using the evaluate_pde tool."""

    # Run AutoGen conversation
    chat_result = self.executor_agent.initiate_chat(
        self.generator_agent,
        message=prompt,
        max_turns=samples_per_prompt * 2  # Allow multiple tool calls
    )

    # Check convergence
    if self.best_score >= convergence_threshold * 10:
        break
```

### 4. Generator Agent Behavior

The `PDE_Generator` agent:
1. Reads the prompt with data summary and top-5 previous PDEs
2. Proposes a novel PDE candidate
3. Calls `evaluate_pde(equation="...")`
4. Receives tool response with score and metrics
5. Analyzes results and proposes next candidate
6. Repeats until `max_turns` reached

---

## 🎓 Ground Truth Challenge

**Target PDE:**
```
∂g/∂t = α·Δg - β·∇·(g∇(ln S)) + γ·g(1-g/K)
```

**True Parameters:**
- α = 0.5   (diffusion coefficient)
- β = 1.5   (chemotaxis coefficient)
- γ = 0.15  (growth rate)
- K = 3.0   (carrying capacity)

**Dynamics:**
- 15.1% mass increase over time
- 1.13x peak density growth
- Strong chemotaxis toward attractant sources
- Logistic growth with carrying capacity

**Difficulty:** HARD
- 4 parameters to discover
- 3 different PDE terms (diffusion, chemotaxis, growth)
- Nonlinear operator ∇·(g∇(ln S))
- Parameter coupling

---

## 📈 Expected Progress

**Iterations 1-100:** Random exploration, finding basic forms
- Expected R² ≈ 0.70 (basic diffusion only)

**Iterations 100-500:** Discovering chemotaxis term
- Expected R² ≈ 0.85 (diffusion + chemotaxis)

**Iterations 500-2000:** Refining parameters, adding growth
- Expected R² ≈ 0.92 (all 3 terms)

**Iterations 2000-5000:** Fine-tuning all 4 parameters
- Expected R² ≈ 0.95+ (near-optimal)

**Iterations 5000-8000:** Convergence or plateau
- Expected R² ≈ 0.97-0.98 (excellent match)

---

## 📁 Output Structure

After running, you'll find:

```
logs/pde_discovery_autogen_8k/
├── tensorboard/                    # TensorBoard logs
│   └── events.out.tfevents.*
├── discovery_results.json          # Final results
├── experience_buffer.json          # All attempted PDEs
├── best_iter_000200.png           # Best PDE at iter 200
├── best_iter_000400.png           # Best PDE at iter 400
├── best_iter_000600.png           # And so on...
└── ...
```

---

## 🔍 Checking Results

### View Final Results
```bash
cat logs/pde_discovery_autogen_8k/discovery_results.json
```

### View Top-5 Discovered PDEs
```bash
/home/gaoch/miniconda3/envs/llmsr/bin/python -c "
import json

with open('logs/pde_discovery_autogen_8k/experience_buffer.json') as f:
    data = json.load(f)

# Sort by score
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

## 🐛 Troubleshooting

### If AutoGen fails to call tool:
- Check that vLLM server supports function calling
- Ensure model is Qwen3-VL-8B-Instruct (supports tools)
- Verify `llm_config` has correct `base_url` and `model`

### If TensorBoard not working:
```bash
pip install tensorboard torch torchvision pillow
```

### If evaluation is too slow:
- Reduce `--max_iterations` to 2000
- Reduce `--samples_per_prompt` to 2
- Use smaller grid size in test data

### Check LLM is running:
```bash
curl http://localhost:10005/v1/models
```

### Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

---

## ✅ Differences from Original Implementation

| Feature | Original (`run_pde_discovery_final.py`) | AutoGen Tool Use (`run_pde_discovery_autogen.py`) |
|---------|----------------------------------------|---------------------------------------------------|
| AutoGen API | ❌ Wrong (autogen 0.7.5) | ✅ Correct (pyautogen 0.2) |
| Tool Use | ❌ No | ✅ Yes (`evaluate_pde` registered) |
| Agent Pattern | Manual LLM calls | ConversableAgent conversations |
| PDE Evaluation | Called directly | Called as tool by Generator agent |
| Fallback | Falls back to direct LLM | Requires AutoGen (no fallback) |
| Config | `llm_config` dict | `llm_config` with `config_list` |

---

## 🎉 Ready to Run!

Everything is set up with the correct AutoGen API and tool use pattern!

**Start Discovery:**
```bash
./RUN_AUTOGEN_DISCOVERY.sh
```

**Monitor in Another Terminal:**
```bash
/home/gaoch/miniconda3/envs/llmsr/bin/tensorboard \
  --logdir logs/pde_discovery_autogen_8k/tensorboard \
  --port 6006
```

**Open Browser:**
```
http://localhost:6006
```

---

**Good luck discovering the PDE with AutoGen! 🚀**
