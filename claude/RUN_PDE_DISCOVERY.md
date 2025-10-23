# Running PDE Discovery - Final Instructions

## ✅ Setup Complete

All components are ready:
- ✅ Complex test case created with dynamic behavior (15% mass increase)
- ✅ TensorBoard logging integrated
- ✅ AutoGen dual-agent system working
- ✅ Experience buffer with in-context learning
- ✅ Comprehensive visualization suite

---

## 📊 Test Data Created

**Dataset**: `logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5`

**Ground Truth PDE**:
```
∂g/∂t = α·Δg - β·∇·(g∇(ln S)) + γ·g(1-g/K)
```

**True Parameters**:
- α = 0.5 (diffusion coefficient)
- β = 1.5 (chemotaxis coefficient)
- γ = 0.15 (growth rate)
- K = 3.0 (carrying capacity)

**Dynamics**:
- 15.1% mass increase over time
- 1.13x peak density growth
- Strong chemotaxis toward attractant sources
- Logistic growth with carrying capacity

---

## 🚀 Running Commands

### Option 1: Full 8000 Iteration Run (Recommended)

```bash
/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_final.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --api_base http://localhost:10005/v1 \
  --api_model /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --max_iterations 8000 \
  --samples_per_prompt 4 \
  --use_autogen \
  --output_dir logs/pde_discovery_8k_final
```

**Features**:
- ✅ 8000 iterations for thorough search
- ✅ AutoGen dual-agent system (--use_autogen flag)
- ✅ TensorBoard logging (auto-enabled)
- ✅ 4 samples per LLM call
- ✅ Saves visualizations every 200 iterations
- ✅ Experience buffer with top-5 context

**Estimated Time**: 8-12 hours (depending on LLM speed)

---

### Option 2: Quick Test Run (500 iterations)

```bash
/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_final.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --api_base http://localhost:10005/v1 \
  --api_model /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --max_iterations 500 \
  --samples_per_prompt 4 \
  --use_autogen \
  --output_dir logs/pde_discovery_test
```

**Estimated Time**: 30-60 minutes

---

### Option 3: Background Run with Logging

```bash
nohup /home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_final.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --api_base http://localhost:10005/v1 \
  --api_model /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --max_iterations 8000 \
  --samples_per_prompt 4 \
  --use_autogen \
  --output_dir logs/pde_discovery_8k_final \
  > logs/pde_discovery_8k_final.log 2>&1 &

# Get process ID
echo $!

# Monitor progress
tail -f logs/pde_discovery_8k_final.log
```

---

## 📈 Monitoring with TensorBoard

While discovery is running, launch TensorBoard:

```bash
/home/gaoch/miniconda3/envs/llmsr/bin/tensorboard --logdir logs/pde_discovery_8k_final/tensorboard --port 6006
```

Then open in browser: `http://localhost:6006`

**TensorBoard Metrics**:
- `metrics/score`: Overall PDE quality score (0-10)
- `metrics/r2`: R² coefficient (goodness of fit)
- `metrics/mse`: Mean squared error
- `metrics/mass_error`: Mass conservation error (%)
- `best/score`: Best score so far
- `best/r2`: Best R² so far
- `visualizations/best`: Best PDE visualizations
- `performance/iteration_time`: Time per iteration
- `performance/buffer_size`: Experience buffer growth

---

## 📁 Output Structure

After running, you'll find:

```
logs/pde_discovery_8k_final/
├── tensorboard/              # TensorBoard logs
│   └── events.out.tfevents.*
├── discovery_results.json    # Final results
├── experience_buffer.json    # All attempted PDEs
├── best_iter_000200.png     # Best PDE at iter 200
├── best_iter_000400.png     # Best PDE at iter 400
├── best_iter_000600.png     # And so on...
└── ...
```

---

## 🔍 Checking Results

### View Final Results
```bash
cat logs/pde_discovery_8k_final/discovery_results.json
```

### Check Best PDEs from Buffer
```bash
/home/gaoch/miniconda3/envs/llmsr/bin/python -c "
import json
with open('logs/pde_discovery_8k_final/experience_buffer.json') as f:
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

## ⚙️ Command Line Arguments

```
--dataset PATH             Path to HDF5 dataset (required)
--api_base URL            vLLM API base URL (default: http://localhost:10005/v1)
--api_model PATH          Model name/path (default: /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct)
--max_iterations N        Maximum iterations (default: 8000)
--samples_per_prompt N    LLM samples per prompt (default: 4)
--use_autogen            Enable AutoGen agents (recommended)
--output_dir PATH         Output directory (default: ./logs/pde_discovery_final)
```

---

## 🎯 Success Criteria

The system will stop when:
1. **Convergence**: Score ≥ 9.8/10 (98% of maximum)
2. **Plateau**: No improvement for 100 iterations
3. **Max iterations**: Reaches 8000 iterations

**Expected Outcomes**:
- Should discover at least 3 of 4 parameters (α, β, γ, K)
- R² should exceed 0.95 for best PDE
- Mass conservation error < 5%
- Visual match between predicted and observed dynamics

---

## 🐛 Troubleshooting

### If AutoGen fails:
```bash
# Run without AutoGen (direct LLM)
/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_final.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --max_iterations 8000 \
  --output_dir logs/pde_discovery_8k_final
# (omit --use_autogen flag)
```

### If TensorBoard not available:
```bash
pip install tensorboard torch torchvision
```

### Check LLM is running:
```bash
curl http://localhost:10005/v1/models
```

### Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

---

## 📊 Expected Progress

**Iterations 1-100**: Initial exploration, random PDEs
**Iterations 100-500**: Finding basic forms (diffusion + chemotaxis)
**Iterations 500-2000**: Refining parameters, discovering growth term
**Iterations 2000-5000**: Fine-tuning all 4 parameters
**Iterations 5000-8000**: Convergence or plateau

**Best R² Timeline**:
- Iter 100: R² ≈ 0.70 (basic diffusion)
- Iter 500: R² ≈ 0.85 (with chemotaxis)
- Iter 2000: R² ≈ 0.92 (with growth)
- Iter 5000: R² ≈ 0.95+ (all parameters)

---

## 🎓 Understanding the Challenge

**Why is this hard?**
1. **4 parameters** to discover (α, β, γ, K)
2. **3 different terms**: diffusion, chemotaxis, logistic growth
3. **Nonlinear operators**: ∇·(g∇(ln S)) is complex
4. **Parameter coupling**: All terms interact
5. **Search space**: Infinite possible PDE forms

**What makes it tractable?**
1. **Experience buffer**: Learns from previous attempts
2. **In-context learning**: Top-5 PDEs guide generation
3. **LLM domain knowledge**: Understands chemotaxis biology
4. **Iterative refinement**: 8000 attempts to converge
5. **Numerical optimization**: Fits parameters for each PDE form

---

## ✅ Final Checklist

Before running:
- [ ] vLLM server is running on port 10005
- [ ] AutoGen is installed: `pip list | grep autogen`
- [ ] TensorBoard is installed: `pip list | grep tensorboard`
- [ ] Complex test data exists: `ls logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5`
- [ ] Enough disk space: ~1GB for outputs
- [ ] Enough time: 8-12 hours for full run

---

## 🚀 RECOMMENDED COMMAND TO RUN NOW

```bash
/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_final.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --api_base http://localhost:10005/v1 \
  --api_model /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --max_iterations 8000 \
  --samples_per_prompt 4 \
  --use_autogen \
  --output_dir logs/pde_discovery_8k_final
```

**Monitor in another terminal:**
```bash
# Watch TensorBoard
/home/gaoch/miniconda3/envs/llmsr/bin/tensorboard --logdir logs/pde_discovery_8k_final/tensorboard --port 6006

# Or tail the output
tail -f logs/pde_discovery_8k_final.log  # (if running in background)
```

---

**Good luck! The system should discover the full PDE within 5000-8000 iterations! 🎉**
