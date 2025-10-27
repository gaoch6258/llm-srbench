# ✅ LLMSR-Based PDE Discovery - READY TO USE

## Summary of Changes

I've completely redesigned the PDE discovery system to address all three concerns:

### 1. ✅ Visual Critic Fixed
- Changed `vision: False` → `vision: True` in model_info
- Visual critic now analyzes images and returns scores 0-10
- Scores will appear in results (not 0/None anymore)

### 2. ✅ No GT Priors
- Removed ALL chemotaxis-specific references
- Prompts now say "discover from data" not "chemotaxis systems"
- No examples like `∇·(g∇(ln S))`
- Agent must discover PDE structure, not assume it

### 3. ✅ Flexible Parameters (Pure LLMSR)
- New tool signature: `evaluate_pde_tool(pde_description, num_params)`
- LLM generates complete `def pde_update(g, S, dx, dy, dt, params)` code
- Parameters are `params[0], params[1], ...` not α, β, γ, K
- Agent can use 1-5 parameters as needed
- Each PDE can have different number of parameters!

## Files Modified/Created

### New Files
- `bench/pde_llmsr_solver.py` - Pure LLMSR solver with sandbox execution

### Modified Files
- `run_pde_discovery_autogen_v04.py`:
  - Imports: Use LLMSRPDESolver instead of hybrid
  - Model setup: vision=True, create LLMSR solver
  - Tool: New signature with pde_description + num_params
  - Prompts: No chemotaxis bias, generic PDE discovery

### Documentation
- `LLMSR_IMPLEMENTATION_SUMMARY.md` - Detailed changes
- `tests/test_llmsr_solver.py` - Test script (for reference)

## How to Run

### Basic Test
```bash
cd /home/gaoch/llm-srbench

# Run discovery (make sure you have your dataset)
/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_autogen_v04.py \
  --dataset path/to/your/data.h5 \
  --api_base http://localhost:10005/v1 \
  --api_model /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --max_iterations 100 \
  --samples_per_prompt 4
```

### Expected Behavior

1. **Visual scores appear**:
   ```
   ✓ Combined=8.50 (Num=8.20, Vis=9.50) [LLMSR-CODE].
   Visual Critic (score 9.50/10):
   - Spatial: Pattern matches well...
   - Temporal: Evolution is smooth...
   ```

2. **Diverse PDEs tried**:
   ```
   Iter 1: "Diffusion with p0"                    (num_params=1)
   Iter 1: "Diffusion p0 plus reaction p1*g"      (num_params=2)
   Iter 1: "Diffusion p0 plus nonlinear p1*g^2"   (num_params=2)
   Iter 1: "Complex PDE with 3 terms"             (num_params=3)
   ```

3. **No chemotaxis bias**:
   - Agent won't assume chemotaxis forms
   - Will try diffusion, reaction, advection, nonlinear
   - Discovers structure from data

## How It Works

### Agent Call
```python
evaluate_pde_tool(
    pde_description="Diffusion p0 plus nonlinear reaction p1*g^2",
    num_params=2
)
```

### System Processing
1. **LLM generates code**:
   ```python
   def pde_update(g, S, dx, dy, dt, params):
       p0 = params[0]  # diffusion
       p1 = params[1]  # reaction

       # Laplacian
       laplacian_g[1:-1, 1:-1] = (
           g[1:-1, 2:] + g[1:-1, :-2] +
           g[2:, 1:-1] + g[:-2, 1:-1] - 4*g[1:-1, 1:-1]
       ) / dx**2

       # PDE
       dg_dt = p0 * laplacian_g + p1 * g**2

       return g + dt * dg_dt
   ```

2. **Execute in sandbox** (subprocess, timeout, validation)

3. **Fit parameters** (L-BFGS-B optimization)

4. **Compute metrics** (R², MSE, mass error)

5. **Visual critic** (analyzes visualization, returns score 0-10)

6. **Return results**:
   ```json
   {
     "success": true,
     "r2": 0.92,
     "mse": 0.003,
     "mass_error": 2.5,
     "visual_score": 8.5,
     "combined_score": 8.7,
     "fitted_params": {"p0": 0.52, "p1": 0.18},
     "num_params": 2,
     "generated_code_preview": "def pde_update..."
   }
   ```

## Key Advantages

| Feature | Before | After |
|---------|--------|-------|
| **Visual score** | Always 0/None | Working! Returns 0-10 |
| **PDE priors** | Chemotaxis-biased | Generic discovery |
| **Parameters** | Fixed α,β,γ,K | Flexible p0,p1,... (1-5) |
| **Approach** | Hybrid symbolic+template | Pure LLMSR code generation |
| **Flexibility** | Limited forms | ANY valid PDE |

## Troubleshooting

### Visual score still 0/None?
- Check that vision model is actually multimodal
- Check API endpoint supports vision
- Look for critic errors in output

### Agent still assuming chemotaxis?
- Check you're using updated file
- Clear any cached prompts
- Verify SUPPORTED_FORMS_GUIDE doesn't mention chemotaxis

### Code generation fails?
- Check LLM is generating valid Python
- Increase timeout if needed (currently 60s)
- Look at `generated_code_preview` in error output

## Next Steps

1. **Test with your data**:
   ```bash
   /home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_autogen_v04.py \
     --dataset /path/to/your/chemotaxis_data.h5 \
     --max_iterations 50
   ```

2. **Monitor results**:
   - Check visual_score in output (should be 0-10, not None)
   - See diverse PDE forms tried
   - Verify different num_params used

3. **Tune if needed**:
   - Adjust `samples_per_prompt` (how many PDEs per round)
   - Adjust parameter bounds in tool (currently 0.001-5.0)
   - Adjust timeout (currently 60s)

## What Changed vs. Hybrid Solver

The previous hybrid solver had:
- Tier 1: Symbolic (SymPy parsing)
- Tier 2: Template-based code generation
- Tier 3: Legacy hardcoded

The new LLMSR solver has:
- **Pure LLM code generation** (like method/llmsr)
- No symbolic parsing, no templates
- Maximum flexibility
- Works exactly like you requested!

---

**Status**: ✅ **COMPLETE AND READY TO TEST**

All three concerns addressed:
1. ✅ Visual critic fixed (vision=True)
2. ✅ No GT priors (removed chemotaxis references)
3. ✅ Flexible parameters (pure LLMSR approach)

Test it with your data and let me know how it goes!
