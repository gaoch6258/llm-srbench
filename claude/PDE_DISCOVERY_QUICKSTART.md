# PDE Discovery - Quick Start Guide

## 🚀 Quick Start (5 minutes)

### 1. Run Tests
```bash
python test_pde_discovery.py
```

This verifies all components work correctly.

### 2. Run Examples
```bash
python example_pde_discovery.py
```

This demonstrates end-to-end usage of all components.

### 3. Try Your Own Data
```python
from bench.pde_datamodule import ChemotaxisProblem
from bench.pde_agents import SimplePDEDiscoverySystem

# Load your data
problem = ChemotaxisProblem(
    g_init=your_initial_density,      # (H, W)
    S=your_chemoattractant,            # (H, W) or (H, W, T)
    g_observed=your_observed_evolution, # (H, W, T)
    metadata={'dx': 1.0, 'dy': 1.0, 'dt': 0.01},
    gt_equation="∂g/∂t = α·Δg - ∇·(g∇(ln S))"  # optional
)

# Run discovery
system = SimplePDEDiscoverySystem(work_dir="./output")
results = system.discover(
    problem.g_init, problem.S, problem.g_observed, verbose=True
)

print(f"Best equation: {results['best_equation']}")
```

## 📦 What's Included

### Core Components

| File | Purpose |
|------|---------|
| `bench/pde_solver.py` | PDE solver with finite differences |
| `bench/pde_visualization.py` | Comprehensive plotting for visual analysis |
| `bench/pde_experience_buffer.py` | Memory system for discovered equations |
| `bench/pde_prompts.py` | Domain-specific prompts for chemotaxis |
| `bench/pde_agents.py` | Dual-agent system with AutoGen |
| `bench/pde_datamodule.py` | Data loading and management |

### Scripts

| File | Purpose |
|------|---------|
| `test_pde_discovery.py` | Comprehensive test suite |
| `example_pde_discovery.py` | End-to-end usage examples |

### Documentation

| File | Purpose |
|------|---------|
| `PDE_DISCOVERY_README.md` | Complete documentation |
| `PDE_DISCOVERY_QUICKSTART.md` | This file |

## 🎯 Use Cases

### Use Case 1: Test with Synthetic Data
```bash
# Generate test data and run discovery
python test_pde_discovery.py
```

### Use Case 2: Analyze Existing PDE
```python
from bench.pde_solver import PDESolver, PDEConfig
import numpy as np

solver = PDESolver(PDEConfig(dx=1.0, dy=1.0, dt=0.01))

# Your initial condition and chemoattractant
H, W = 256, 256
g_init = ...  # Your initial density
S = ...       # Your chemoattractant field

# Solve reference PDE
g_solution = solver.solve_reference_pde(g_init, S, alpha=0.5, num_steps=100)

# Compare with observations
mse = solver.compute_spatiotemporal_loss(g_solution, g_observed, 'mse')
r2 = solver.compute_spatiotemporal_loss(g_solution, g_observed, 'r2')
```

### Use Case 3: Visualize Results
```python
from bench.pde_visualization import PDEVisualizer

visualizer = PDEVisualizer()

# Create comprehensive analysis
img = visualizer.create_critique_visualization(
    observed=g_observed,
    predicted=g_predicted,
    equation_str="∂g/∂t = α·Δg - ∇·(g∇(ln S))",
    metrics={'mse': 0.001, 'r2': 0.95},
    save_path="analysis.png"
)
```

### Use Case 4: Track Discovery Progress
```python
from bench.pde_experience_buffer import PDEExperienceBuffer

buffer = PDEExperienceBuffer(max_size=100)

# During discovery loop
for iteration in range(max_iterations):
    # ... evaluate candidate equation ...

    buffer.add(
        equation=candidate_equation,
        score=score,
        metrics=metrics,
        visual_analysis=visual_analysis,
        reasoning=reasoning,
        suggestions=suggestions
    )

    # Get context for next iteration
    context = buffer.format_for_prompt(k=5)

# Review best attempts
best = buffer.get_best()
top_5 = buffer.get_top_k(k=5)
```

### Use Case 5: Full Discovery with AutoGen
```python
from bench.pde_agents import PDEDiscoverySystem

# Requires: pip install pyautogen
# Requires: vLLM server running with Qwen2-VL-7B-Instruct

system = PDEDiscoverySystem(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    api_base="http://localhost:8000/v1",
    max_iterations=20,
    convergence_threshold=0.95
)

results = system.discover(
    g_init=problem.g_init,
    S=problem.S,
    g_observed=problem.g_observed,
    problem_description="Neutrophil chemotaxis",
    verbose=True
)
```

## 📊 Expected Outputs

### Test Suite Output
```
============================================================
PDE DISCOVERY SYSTEM - COMPREHENSIVE TEST
============================================================

============================================================
TEST 1: PDE Solver
============================================================
✓ Solution shape: (32, 32, 30)
✓ Mass conservation: -0.15%
✓ Fitted α: 0.4982 (true: 0.5000)

============================================================
TEST 2: Visualization
============================================================
✓ Comprehensive plot created: test_comprehensive.png
✓ Critique plot created: test_critique.png

...

ALL TESTS PASSED! ✓
```

### Generated Files

After running tests and examples:
```
test_comprehensive.png          # Multi-panel visualization
test_critique.png              # Focused critique visualization
test_buffer.json               # Experience buffer state
test_chemotaxis.hdf5          # Test dataset
example_comprehensive.png      # Example visualization
example_critique.png          # Example critique
example_buffer.json           # Example buffer
example_chemotaxis.hdf5       # Example dataset
example_discovery/            # Discovery output directory
```

## 🔧 Common Workflows

### Workflow 1: Evaluate Known PDE
```python
# 1. Load data
from bench.pde_datamodule import ChemotaxisDataModule
dm = ChemotaxisDataModule(data_source="hdf5", data_path="data.hdf5")
problem = list(dm.load().values())[0]

# 2. Solve with known equation
from bench.pde_solver import PDESolver
solver = PDESolver()
predicted, info = solver.evaluate_pde(
    "∂g/∂t = α·Δg - ∇·(g∇(ln S))",
    problem.g_init,
    problem.S,
    param_values={'α': 0.5},
    num_steps=problem.g_observed.shape[2]
)

# 3. Visualize
from bench.pde_visualization import PDEVisualizer
visualizer = PDEVisualizer()
metrics = {
    'mse': solver.compute_spatiotemporal_loss(predicted, problem.g_observed, 'mse'),
    'r2': solver.compute_spatiotemporal_loss(predicted, problem.g_observed, 'r2')
}
visualizer.create_critique_visualization(
    problem.g_observed, predicted, "∂g/∂t = α·Δg - ∇·(g∇(ln S))",
    metrics, save_path="evaluation.png"
)
```

### Workflow 2: Fit PDE Parameters
```python
from bench.pde_solver import PDESolver

solver = PDESolver()

# Define PDE template with parameters to fit
pde_template = "∂g/∂t = α·Δg - ∇·(g∇(ln S))"

# Fit parameters
fitted_params, loss = solver.fit_pde_parameters(
    pde_template,
    g_init=problem.g_init,
    S=problem.S,
    g_observed=problem.g_observed,
    param_bounds={'α': (0.01, 2.0)},
    method='L-BFGS-B'
)

print(f"Fitted α: {fitted_params['α']:.4f}")
print(f"Loss: {loss:.6f}")
```

### Workflow 3: Discover New PDE
```python
from bench.pde_agents import SimplePDEDiscoverySystem

# Simple version (tests predefined candidates)
system = SimplePDEDiscoverySystem(max_iterations=10)
results = system.discover(g_init, S, g_observed, verbose=True)

# Full AI-driven version (requires AutoGen + vLLM)
from bench.pde_agents import PDEDiscoverySystem
system = PDEDiscoverySystem(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    api_base="http://localhost:8000/v1"
)
results = system.discover(g_init, S, g_observed, verbose=True)
```

## 🎓 Learning Path

### Beginner
1. Run `test_pde_discovery.py` to understand components
2. Read through `example_pde_discovery.py`
3. Modify examples to use your own data
4. Experiment with different PDE equations

### Intermediate
1. Create custom visualizations in `pde_visualization.py`
2. Add new PDE terms in `pde_solver.py`
3. Customize prompts in `pde_prompts.py`
4. Experiment with different solver configurations

### Advanced
1. Implement implicit PDE solvers
2. Add 3D spatiotemporal support
3. Create custom agent strategies
4. Integrate with experimental data pipelines
5. Optimize performance for large datasets

## 🐛 Troubleshooting

### Issue: Import errors
```bash
# Install missing dependencies
pip install numpy scipy matplotlib pillow h5py
pip install pyautogen  # For full agent system
```

### Issue: CFL condition violations
```python
# Reduce time step or increase spatial resolution
config = PDEConfig(dt=0.001, dx=0.5, dy=0.5)
solver = PDESolver(config)
```

### Issue: AutoGen not available
```python
# Use simplified version
from bench.pde_agents import SimplePDEDiscoverySystem
system = SimplePDEDiscoverySystem()
```

### Issue: Numerical instability
```python
# Check stability and adjust parameters
solver.config.stability_check = True
solver.config.diffusion_limit = 0.4  # More conservative
```

## 📚 Next Steps

1. **Read Full Documentation**: See `PDE_DISCOVERY_README.md`
2. **Explore Examples**: Run `python example_pde_discovery.py --example N`
3. **Try Your Data**: Follow Workflow 3 above
4. **Contribute**: Add new PDE terms, solvers, or visualizations

## 💡 Tips

- Start with small grids (64×64) for faster iteration
- Use visualization extensively to understand PDE behavior
- Experience buffer is key for in-context learning
- Adjust convergence thresholds based on data noise level
- Save intermediate results frequently

## 🤝 Support

For questions or issues:
1. Check `PDE_DISCOVERY_README.md` for detailed docs
2. Review inline code comments
3. Run test suite to verify setup
4. File issues on GitHub repository

---

**Ready to discover PDEs? Start with:**
```bash
python test_pde_discovery.py
python example_pde_discovery.py
```
