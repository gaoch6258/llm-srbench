# PDE Discovery Extension for LLM-SR

This extension adds spatiotemporal PDE discovery capabilities to the LLM-SR benchmark, specifically targeting neutrophil chemotaxis from imaging data.

## Overview

The PDE Discovery system discovers partial differential equations from spatiotemporal imaging data using a dual-agent approach with vision-enabled LLMs.

### Key Features

- **PDE Solver**: Numerical solver for 2D+time PDEs with chemotaxis and diffusion terms
- **Dual-Agent System**: Generator + Visual Critic using Qwen3-VL-8B-Instruct
- **Experience Buffer**: Memory system for in-context learning
- **Rich Visualization**: Comprehensive plots for visual analysis
- **Modular Design**: Integrates seamlessly with existing LLM-SR architecture

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PDE Discovery System                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐        ┌──────────────┐                  │
│  │   Equation   │◄──────►│   Visual     │                  │
│  │  Generator   │        │   Critic     │                  │
│  └──────┬───────┘        └──────▲───────┘                  │
│         │                       │                           │
│         │                       │                           │
│         ▼                       │                           │
│  ┌──────────────┐        ┌─────┴────────┐                 │
│  │     PDE      │        │ Visualization│                 │
│  │   Solver     │───────►│    Suite     │                 │
│  └──────────────┘        └──────────────┘                 │
│         │                                                   │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │  Experience  │                                          │
│  │   Buffer     │                                          │
│  └──────────────┘                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. PDE Solver (`bench/pde_solver.py`)

Numerical solver for 2D+time PDEs supporting:

- **Operators**: ∇ (gradient), ∇· (divergence), Δ (Laplacian), ∂/∂t (time derivative)
- **Terms**: Diffusion (α·Δg), Chemotaxis (∇·(g∇(ln S))), custom combinations
- **Boundary Conditions**: Periodic, Neumann, Dirichlet
- **Stability**: CFL condition checking for explicit schemes
- **Parameter Fitting**: Scipy-based optimization to fit PDE parameters

**Key Classes:**
- `PDESolver`: Main solver class
- `PDEConfig`: Configuration dataclass

**Example:**
```python
from bench.pde_solver import PDESolver, PDEConfig

solver = PDESolver(PDEConfig(dx=1.0, dy=1.0, dt=0.01))
g_history = solver.solve_reference_pde(g_init, S, alpha=0.5, num_steps=100)
```

### 2. Visualization Suite (`bench/pde_visualization.py`)

Comprehensive visualization for visual analysis:

- **Temporal snapshots**: Early/mid/late comparisons
- **Difference maps**: Spatial error distribution
- **Temporal evolution**: Mean/std/mass over time
- **Gradient fields**: Vector plots of spatial derivatives
- **Fourier spectra**: Frequency domain analysis
- **Conservation checks**: Mass/energy plots

**Key Classes:**
- `PDEVisualizer`: Main visualization class

**Example:**
```python
from bench.pde_visualization import PDEVisualizer

visualizer = PDEVisualizer()
img = visualizer.create_critique_visualization(
    observed, predicted, equation_str, metrics
)
```

### 3. Experience Buffer (`bench/pde_experience_buffer.py`)

Memory system storing (equation, score, analysis, reasoning) tuples:

- **Storage**: Persistent JSON-based storage
- **Retrieval**: Top-K by score, recent N, best/worst
- **Diversity**: Edit distance-based pruning
- **Prompt Formatting**: Automatic context generation for in-context learning

**Key Classes:**
- `PDEExperienceBuffer`: Buffer management
- `PDEExperience`: Single experience entry

**Example:**
```python
from bench.pde_experience_buffer import PDEExperienceBuffer

buffer = PDEExperienceBuffer(max_size=100)
buffer.add(equation, score, metrics, visual_analysis, reasoning, suggestions)
top_5 = buffer.get_top_k(k=5)
```

### 4. Prompt System (`bench/pde_prompts.py`)

Domain-specific prompts for chemotaxis:

- **Operator Library**: Physical interpretations of PDE operators
- **Context**: Neutrophil biology and chemotaxis mechanisms
- **Constraints**: Conservation laws, dimensional consistency, physical bounds
- **Structured Output**: XML-like tags for parsing responses

**Key Functions:**
- `create_generator_prompt()`: For Equation Generator agent
- `create_critic_prompt()`: For Visual Critic agent
- `extract_*()`: Parsing utilities for structured responses

### 5. Dual-Agent System (`bench/pde_agents.py`)

AutoGen-based multi-agent orchestration:

- **Equation Generator**: Proposes PDE candidates using domain knowledge
- **Visual Critic**: Analyzes visualizations and provides scores/feedback
- **Convergence**: Score thresholds and plateau detection
- **Fallback**: Simplified version without AutoGen for testing

**Key Classes:**
- `PDEDiscoverySystem`: Full AutoGen-based system
- `SimplePDEDiscoverySystem`: Testing/fallback version

**Example:**
```python
from bench.pde_agents import PDEDiscoverySystem

system = PDEDiscoverySystem(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    api_base="http://localhost:8000/v1",
    max_iterations=20
)

results = system.discover(g_init, S, g_observed)
```

### 6. DataModule (`bench/pde_datamodule.py`)

Dataset management compatible with existing pipeline:

- **Synthetic Data**: Generated from reference PDE
- **HDF5 Support**: Load/save experimental data
- **SEDTask Conversion**: Compatible with existing LLM-SR pipeline
- **Test Data Generation**: Utilities for creating test datasets

**Key Classes:**
- `ChemotaxisProblem`: Problem definition
- `ChemotaxisDataModule`: Dataset loader/manager

## Installation

### Prerequisites

```bash
# Core dependencies (already in requirements.txt)
pip install numpy scipy matplotlib pillow h5py

# AutoGen for multi-agent system
pip install pyautogen

# For local LLM serving (optional)
pip install vllm
```

### Setup

1. **Clone Repository** (if not already done)
```bash
cd /home/gaoch/llm-srbench
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
pip install pyautogen
```

3. **Test Installation**
```bash
python test_pde_discovery.py
```

## Usage

### Quick Start: Testing Components

```bash
# Run comprehensive test suite
python test_pde_discovery.py
```

This tests:
- PDE solver accuracy
- Visualization generation
- Experience buffer operations
- Data loading/saving
- Simplified discovery loop

### Using Individual Components

#### 1. Solve a PDE

```python
from bench.pde_solver import PDESolver, PDEConfig
import numpy as np

# Setup
solver = PDESolver(PDEConfig(dx=1.0, dy=1.0, dt=0.01))
H, W, T = 256, 256, 100

# Create initial condition and chemoattractant field
X, Y = np.meshgrid(np.arange(W), np.arange(H))
g_init = np.exp(-((X - W/2)**2 + (Y - H/2)**2) / (0.1 * W**2))
S = np.exp(0.01 * X + 0.01 * Y)

# Solve reference chemotaxis PDE
alpha = 0.5
g_history = solver.solve_reference_pde(g_init, S, alpha, num_steps=T)

# Compute metrics
mse = solver.compute_spatiotemporal_loss(g_history, g_observed, 'mse')
```

#### 2. Generate Visualizations

```python
from bench.pde_visualization import PDEVisualizer

visualizer = PDEVisualizer(figsize=(16, 12))

# Create comprehensive plot
img = visualizer.create_comprehensive_plot(
    observed=g_observed,
    predicted=g_predicted,
    equation_str="∂g/∂t = α·Δg - ∇·(g∇(ln S))",
    score=8.5,
    save_path="analysis.png"
)

# Create critique-focused plot
metrics = {'mse': 0.001, 'r2': 0.95, 'nmse': 0.05}
img = visualizer.create_critique_visualization(
    observed, predicted, equation_str, metrics, save_path="critique.png"
)
```

#### 3. Use Experience Buffer

```python
from bench.pde_experience_buffer import PDEExperienceBuffer

buffer = PDEExperienceBuffer(max_size=100)

# Add experience
buffer.add(
    equation="∂g/∂t = α·Δg - ∇·(g∇(ln S))",
    score=8.5,
    metrics={'mse': 0.001, 'r2': 0.95},
    visual_analysis="Strong spatial accuracy...",
    reasoning="Balanced chemotaxis and diffusion...",
    suggestions="Try varying α...",
    parameters={'α': 0.5}
)

# Retrieve for prompting
top_5 = buffer.get_top_k(k=5)
prompt_context = buffer.format_for_prompt(k=5, include_visual=True)

# Save/load
buffer.save("buffer.json")
loaded = PDEExperienceBuffer.load("buffer.json")
```

#### 4. Load Data

```python
from bench.pde_datamodule import ChemotaxisDataModule

# Synthetic data
dm = ChemotaxisDataModule(data_source="synthetic")
problems = dm.load()

# HDF5 data
dm = ChemotaxisDataModule(data_source="hdf5", data_path="data.hdf5")
problems = dm.load()

# Access problem
problem = problems['problem_001']
task = problem.to_sed_task()  # Convert to SEDTask
```

### Full Discovery Pipeline

#### With AutoGen (Production)

```python
from bench.pde_agents import PDEDiscoverySystem
from bench.pde_datamodule import ChemotaxisDataModule

# Load data
dm = ChemotaxisDataModule(data_source="synthetic")
problems = dm.load()
problem = list(problems.values())[0]

# Initialize discovery system
system = PDEDiscoverySystem(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
    max_iterations=20,
    convergence_threshold=0.95,
    plateau_patience=5,
    work_dir="./discovery_output"
)

# Run discovery
results = system.discover(
    g_init=problem.g_init,
    S=problem.S,
    g_observed=problem.g_observed,
    problem_description="Neutrophil chemotaxis toward LTB4 gradient",
    verbose=True
)

# Results
print(f"Best equation: {results['best_equation']}")
print(f"Best score: {results['best_score']:.2f}/10")
print(f"Total iterations: {results['total_iterations']}")
```

#### Simplified Version (Testing)

```python
from bench.pde_agents import SimplePDEDiscoverySystem

system = SimplePDEDiscoverySystem(max_iterations=5, work_dir="./test_output")

results = system.discover(
    g_init=problem.g_init,
    S=problem.S,
    g_observed=problem.g_observed,
    verbose=True
)
```

## Configuration

### PDE Solver Configuration

```python
from bench.pde_solver import PDEConfig

config = PDEConfig(
    dx=1.0,                  # Spatial step in x
    dy=1.0,                  # Spatial step in y
    dt=0.01,                 # Time step
    boundary_condition="periodic",  # 'periodic', 'neumann', 'dirichlet'
    max_iterations=1000,     # Max time steps
    stability_check=True,    # Check CFL condition
    diffusion_limit=0.5      # CFL stability limit
)
```

### Discovery System Configuration

```python
system = PDEDiscoverySystem(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
    max_iterations=20,              # Max discovery iterations
    convergence_threshold=0.95,     # Score threshold (0-1) for early stopping
    plateau_patience=5,             # Iterations without improvement before stop
    solver_config=config,           # PDE solver config
    buffer_size=100,                # Experience buffer size
    work_dir="./output"             # Output directory
)
```

## Data Format

### Input Data Structure

```python
ChemotaxisProblem(
    g_init: np.ndarray,      # Shape (H, W) - initial cell density
    S: np.ndarray,           # Shape (H, W) or (H, W, T) - chemoattractant
    g_observed: np.ndarray,  # Shape (H, W, T) - observed evolution
    metadata: Dict,          # dx, dy, dt, etc.
    gt_equation: str         # Ground truth (optional)
)
```

### HDF5 Format

```
chemotaxis_data.hdf5
├── problem_001/
│   ├── g_init          (H, W)
│   ├── S               (H, W) or (H, W, T)
│   ├── g_observed      (H, W, T)
│   └── attrs:
│       ├── dx, dy, dt
│       ├── gt_equation (optional)
│       └── ...
├── problem_002/
│   └── ...
```

## Extending the System

### Adding New PDE Terms

Edit `bench/pde_solver.py`:

```python
def custom_term(self, g: np.ndarray, S: np.ndarray, **kwargs) -> np.ndarray:
    """Custom PDE term"""
    # Your implementation
    return result

# Then in evaluate_pde():
if 'custom' in parsed['normalized']:
    result = self.custom_term(g_current, S_current)
```

### Adding New Visualizations

Edit `bench/pde_visualization.py`:

```python
def create_custom_plot(self, observed, predicted, **kwargs):
    """Custom visualization"""
    fig, ax = plt.subplots()
    # Your plotting code
    return self._fig_to_image(fig)
```

### Customizing Prompts

Edit `bench/pde_prompts.py`:

```python
def create_custom_prompt(context: Dict) -> str:
    """Custom prompt for specific domain"""
    return f"""
    Your custom prompt template
    {context['key']}
    """
```

## Integration with Existing LLM-SR Pipeline

The PDE discovery components integrate seamlessly:

### 1. Add PDE Searcher

Create `methods/pde_sr/searcher.py`:

```python
from bench.searchers.base import BaseSearcher
from bench.pde_agents import PDEDiscoverySystem

class PDESearcher(BaseSearcher):
    def __init__(self, config):
        self.system = PDEDiscoverySystem(**config)

    def discover(self, task: SEDTask) -> List[SearchResult]:
        # Extract spatiotemporal data from task
        samples = task.samples
        results = self.system.discover(
            samples['g_init'],
            samples['S'],
            samples['g_observed']
        )

        # Convert to SearchResult format
        return [SearchResult(
            equation=results['best_equation'],
            score=results['best_score'],
            # ... other fields
        )]
```

### 2. Add Configuration

Create `configs/pde_sr_qwen_vl.yaml`:

```yaml
name: PDE-SR-Qwen-VL
class_name: PDESearcher
api_type: "vllm"
api_model: "Qwen/Qwen2-VL-7B-Instruct"
api_url: "http://localhost:8000/v1/"
max_iterations: 20
convergence_threshold: 0.95
```

### 3. Run Evaluation

```bash
python eval.py --config configs/pde_sr_qwen_vl.yaml --dataset chemotaxis
```

## Troubleshooting

### Common Issues

#### 1. CFL Condition Violations

**Error**: `Warning: CFL condition violated`

**Solution**: Reduce time step or increase spatial resolution:
```python
config = PDEConfig(dt=0.001, dx=0.5, dy=0.5)
```

#### 2. AutoGen Import Errors

**Error**: `ImportError: No module named 'autogen'`

**Solution**: Install AutoGen:
```bash
pip install pyautogen
```

Or use simplified version:
```python
from bench.pde_agents import SimplePDEDiscoverySystem
```

#### 3. Numerical Instability

**Error**: Solution explodes (NaN/Inf values)

**Solutions**:
- Check CFL condition
- Reduce time step
- Adjust diffusion coefficient bounds
- Use implicit schemes (future work)

#### 4. Poor Convergence

**Issue**: Discovery doesn't find good PDEs

**Solutions**:
- Increase max_iterations
- Adjust prompt templates in `pde_prompts.py`
- Expand operator library
- Increase buffer size for more context

## Performance Considerations

### Computational Cost

- **PDE Solving**: O(H × W × T) per iteration
- **Visualization**: O(H × W) per plot
- **LLM Calls**: Dominant cost (~1-5s per call)

### Optimization Tips

1. **Reduce Grid Size**: Use 64×64 or 128×128 for development
2. **Fewer Timepoints**: Sample every Nth frame
3. **Parallel Evaluation**: Test multiple equations concurrently
4. **Caching**: Reuse solved PDEs with same parameters

### Typical Runtime

- **Small problem** (64×64, T=50): ~10-30 seconds per iteration
- **Medium problem** (128×128, T=100): ~30-60 seconds per iteration
- **Large problem** (256×256, T=200): ~2-5 minutes per iteration

Full discovery (20 iterations): 10-90 minutes depending on problem size

## Testing

Run test suite:
```bash
python test_pde_discovery.py
```

Expected output:
```
============================================================
PDE DISCOVERY SYSTEM - COMPREHENSIVE TEST
============================================================

============================================================
TEST 1: PDE Solver
============================================================
✓ Solution shape: (32, 32, 30)
✓ Mass conservation: -0.15%
...

ALL TESTS PASSED! ✓
```

## Citation

If you use this PDE discovery extension, please cite:

```bibtex
@software{pde_discovery_llm_sr,
  title={PDE Discovery Extension for LLM-SR},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/llm-srbench}
}
```

## Future Enhancements

Planned improvements:

1. **Advanced Solvers**: Implicit schemes, adaptive time-stepping
2. **More Operators**: Higher-order derivatives, nonlocal terms
3. **3D Support**: Extend to 3D spatiotemporal data
4. **Real Data**: Integration with experimental imaging datasets
5. **Bayesian Optimization**: Smarter parameter search
6. **Uncertainty Quantification**: Confidence bounds on discovered PDEs
7. **Symbolic Simplification**: Automatic equation simplification
8. **Multi-physics**: Coupled PDE systems

## Support

For issues, questions, or contributions:

- **Issues**: File at GitHub repository
- **Documentation**: See this README and inline code comments
- **Examples**: See `test_pde_discovery.py` for usage examples

## License

Same as parent LLM-SR repository.
