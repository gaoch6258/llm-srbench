# vLLM Tool Calling Issue - Fix Guide

## ❌ Error

```
Error code: 400 - {'error': {'message': '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set', 'type': 'BadRequestError', 'param': None, 'code': 400}}
```

## 🔍 Root Cause

Your vLLM server is missing flags needed for function/tool calling support.

**Current command** (from `engine.sh`):
```bash
vllm serve /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --port 10005 --max-model-len 40730 --tensor-parallel-size 4
```

**Missing flags:**
- `--enable-auto-tool-choice` - Enables automatic tool selection
- `--tool-call-parser hermes` - Specifies tool call parser

---

## ✅ Two Solutions

### **Option 1: Restart vLLM with Tool Support (Recommended for Tool Calling)**

#### Step 1: Stop current vLLM
```bash
# Find the process
ps aux | grep vllm

# Kill it
kill <PID>
```

#### Step 2: Start with tool support
```bash
./engine_with_tools.sh
```

**Contents of `engine_with_tools.sh`:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --dtype auto \
  --port 10005 \
  --max-model-len 40730 \
  --tensor-parallel-size 4 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

#### Step 3: Run discovery with tool calling
```bash
./RUN_AUTOGEN_V04.sh
```

**Pros:**
- ✅ Uses full AutoGen v0.4 tool calling feature
- ✅ Agent can call `evaluate_pde` tool directly
- ✅ More structured interaction

**Cons:**
- ❌ Requires restarting vLLM
- ❌ May have compatibility issues with some models

---

### **Option 2: Use Simple Version (Works NOW, No vLLM Restart)**

#### Run WITHOUT tool calling
```bash
./RUN_SIMPLE_V04.sh
```

**How it works:**
- Agent generates PDE equations as text
- We extract equations using regex patterns
- Evaluate each equation
- Provide feedback in next prompt

**Pros:**
- ✅ Works with current vLLM server immediately
- ✅ No restart needed
- ✅ Still uses AutoGen v0.4 AssistantAgent
- ✅ Still has TensorBoard + experience buffer

**Cons:**
- ❌ No direct tool calling (uses text generation + parsing)
- ❌ Slightly less structured

---

## 📊 Comparison

| Feature | With Tool Calling | Without Tool Calling |
|---------|-------------------|----------------------|
| **vLLM Restart** | Required | Not required |
| **vLLM Flags** | `--enable-auto-tool-choice --tool-call-parser hermes` | None needed |
| **Script** | `run_pde_discovery_autogen_v04.py` | `run_pde_discovery_simple_v04.py` |
| **Run Command** | `./RUN_AUTOGEN_V04.sh` | `./RUN_SIMPLE_V04.sh` |
| **Agent Pattern** | `tools=[evaluate_pde_tool]` | Text generation + parsing |
| **AutoGen v0.4** | ✅ Yes | ✅ Yes |
| **AssistantAgent** | ✅ Yes | ✅ Yes |
| **TensorBoard** | ✅ Yes | ✅ Yes |
| **Experience Buffer** | ✅ Yes | ✅ Yes |
| **Works Now** | ❌ No (needs restart) | ✅ Yes |

---

## 🚀 Quick Start (Option 2 - Immediate)

**Run this NOW without any changes:**

```bash
# Quick test (50 iterations, ~5 min)
/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_simple_v04.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --max_iterations 50 \
  --samples_per_prompt 2 \
  --output_dir logs/pde_test_simple

# Full run (8000 iterations)
./RUN_SIMPLE_V04.sh
```

**Monitor:**
```bash
tensorboard --logdir logs/pde_discovery_simple_v04_8k/tensorboard --port 6006
```

---

## 🔧 If You Want Tool Calling (Option 1)

### Step-by-Step

1. **Kill current vLLM:**
```bash
pkill -f "vllm serve"
```

2. **Start vLLM with tool support:**
```bash
chmod +x engine_with_tools.sh
./engine_with_tools.sh &
```

3. **Wait for vLLM to be ready (~30 seconds):**
```bash
# Test if ready
curl http://localhost:10005/v1/models
```

4. **Run discovery with tool calling:**
```bash
./RUN_AUTOGEN_V04.sh
```

---

## 📝 Which Should You Use?

### Use **Option 2 (Simple, No Tool Calling)** if:
- ✅ You want to start **immediately**
- ✅ You don't want to restart vLLM
- ✅ Text generation + parsing is acceptable

### Use **Option 1 (With Tool Calling)** if:
- ✅ You can restart vLLM
- ✅ You want more structured agent-tool interaction
- ✅ You want to explore full AutoGen v0.4 capabilities

---

## 🎯 Recommendation

**For now:** Use **Option 2** (Simple version)
- Start discovery immediately
- No downtime
- Still gets good results

**Later:** Try **Option 1** (Tool calling)
- Restart vLLM with tool support
- Compare results
- See which works better

---

## 📊 Expected Results (Both Options)

Both should discover the PDE successfully:

**Target:** `∂g/∂t = α·Δg - β·∇·(g∇(ln S)) + γ·g(1-g/K)`

**Expected Progress:**
- Iter 100: R² ≈ 0.70 (diffusion)
- Iter 500: R² ≈ 0.85 (+ chemotaxis)
- Iter 2000: R² ≈ 0.92 (+ growth)
- Iter 5000+: R² ≈ 0.95+ (full PDE)

---

## ✅ Current Status

- ✅ `run_pde_discovery_simple_v04.py` - **READY TO RUN NOW**
- ✅ `RUN_SIMPLE_V04.sh` - **READY TO RUN NOW**
- ✅ `run_pde_discovery_autogen_v04.py` - Ready (needs vLLM restart)
- ✅ `RUN_AUTOGEN_V04.sh` - Ready (needs vLLM restart)
- ✅ `engine_with_tools.sh` - Ready (restart script)

---

**Start discovery NOW with:**
```bash
./RUN_SIMPLE_V04.sh
```

Or restart vLLM first and use tool calling:
```bash
pkill -f "vllm serve"
./engine_with_tools.sh &
sleep 30
./RUN_AUTOGEN_V04.sh
```
