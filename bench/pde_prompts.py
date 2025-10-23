"""
PDE-Specific Prompts for Chemotaxis Domain

This module contains carefully designed prompts for:
1. Equation Generator Agent
2. Visual Critic Agent
"""

from typing import Dict, List, Optional


# Operator library with physical interpretations
OPERATOR_LIBRARY = """
## PDE Operator Library

### Spatial Operators:
- **∇** (gradient): Represents spatial rate of change, points in direction of steepest increase
  - Example: ∇S points toward higher chemoattractant concentration

- **∇·** (divergence): Measures net flow out of a point
  - Example: ∇·J represents net flux of particles

- **Δ** or **∇²** (Laplacian): Second-order diffusion operator
  - Example: Δg represents isotropic diffusion of cell density

### Temporal Operators:
- **∂/∂t**: Time derivative, rate of change over time
  - Example: ∂g/∂t represents how cell density changes

### Common Physical Terms:
- **Diffusion**: α·Δg (random motion, spreading)
- **Chemotaxis**: ∇·(g∇S) or ∇·(g∇(ln S)) (directed motion up/down gradient)
- **Advection**: v·∇g (transport by flow field v)
- **Reaction**: f(g) (growth, decay, interaction)
"""


# Physical context for chemotaxis
CHEMOTAXIS_CONTEXT = """
## Neutrophil Chemotaxis Background

**Biological Context:**
Neutrophils are immune cells that migrate toward sites of infection/injury by following
chemical gradients of chemoattractants like LTB4 (measured via Ca²⁺ staining).

**Key Physical Phenomena:**
1. **Chemotaxis**: Directed migration up chemoattractant gradient
   - Cells sense ∇S and move toward higher concentrations
   - Flux: J_chemotaxis = -χ·g·∇S or -χ·g·∇(ln S)

2. **Random Diffusion**: Brownian motion and cell-cell repulsion
   - Cells undergo random walks
   - Flux: J_diffusion = -α·∇g

3. **Conservation Law**: Mass conservation for cell density
   - ∂g/∂t + ∇·J_total = 0
   - Or: ∂g/∂t = -∇·(J_chemotaxis + J_diffusion)

**Reference Model:**
∇·(g∇(ln S)) = α·Δg - ∂g/∂t

This states: chemotaxis flux = diffusion - accumulation
"""


# Constraints and guidelines
PDE_CONSTRAINTS = """
## PDE Discovery Constraints

### Physical Constraints:
1. **Conservation**: Total cell mass should be approximately conserved
   - Check: ∫∫ g(x,y,t) dx dy ≈ constant

2. **Non-negativity**: Cell density g ≥ 0 everywhere

3. **Dimensional Consistency**: All terms must have same units
   - [∂g/∂t] = cells/(volume·time)
   - [Δg] = cells/(volume·length²)
   - Terms combined must match units

### Mathematical Constraints:
1. **Well-posedness**: PDE should be well-posed (existence, uniqueness, stability)

2. **Order**: Typically second-order in space, first-order in time

3. **Linearity**: Can be linear or nonlinear, but prefer simpler forms initially

### Complexity Guidelines:
- Start with standard forms (diffusion + chemotaxis)
- Add complexity only if simpler forms fail
- Prefer physically interpretable terms
- Avoid overfitting with too many parameters
"""


def create_generator_prompt(
    problem_description: str,
    data_summary: Dict,
    previous_experiences: str,
    iteration: int
) -> str:
    """
    Create prompt for Equation Generator agent

    Args:
        problem_description: Description of the problem
        data_summary: Statistical summary of data
        previous_experiences: Formatted string from experience buffer
        iteration: Current iteration number

    Returns:
        Formatted prompt string
    """
    prompt = f"""# Task: Discover PDE for Neutrophil Chemotaxis

You are an expert in mathematical biology and PDE modeling. Your task is to discover
a partial differential equation (PDE) that describes neutrophil chemotaxis from
spatiotemporal imaging data.

{CHEMOTAXIS_CONTEXT}

{OPERATOR_LIBRARY}

{PDE_CONSTRAINTS}

## Problem Description
{problem_description}

## Data Summary
- Shape: {data_summary.get('shape', 'N/A')}
- Cell density (g) range: [{data_summary.get('g_min', 0):.4f}, {data_summary.get('g_max', 1):.4f}]
- Chemoattractant (S) range: [{data_summary.get('S_min', 0):.4f}, {data_summary.get('S_max', 1):.4f}]
- Timepoints: {data_summary.get('T', 'N/A')}
- Spatial resolution: dx={data_summary.get('dx', 1):.2f}, dy={data_summary.get('dy', 1):.2f}
- Temporal resolution: dt={data_summary.get('dt', 0.01):.4f}

## Statistical Features:
- Mean cell density: {data_summary.get('g_mean', 'N/A'):.4f}
- Std cell density: {data_summary.get('g_std', 'N/A'):.4f}
- Total mass change: {data_summary.get('mass_change_pct', 'N/A'):.2f}%
- Spatial spread: {data_summary.get('spatial_spread', 'N/A')}

{previous_experiences}

## Current Iteration: {iteration}

## Your Task:

Generate a novel PDE that could describe this chemotaxis system. Consider:
1. Previous attempts and what made them succeed/fail
2. Physical plausibility (conservation, non-negativity, dimensional consistency)
3. Balance between chemotaxis and diffusion terms
4. Appropriate functional forms (linear vs nonlinear, logarithmic vs power-law)

## Output Format:

Provide your response in the following structured format:

<equation>
[Write the PDE equation here using operators: ∇, ∇·, Δ, ∂/∂t]
Example: ∂g/∂t = α·Δg - ∇·(g∇(ln S))
</equation>

<reasoning>
[Explain your reasoning for this PDE choice:
- Why this form?
- What physical phenomena does each term represent?
- How does it differ from previous attempts?
- What improvement do you expect?]
</reasoning>

<parameters>
[List parameters to be fitted with suggested ranges:
Example: α (diffusion coefficient): [0.01, 1.0]
         χ (chemotaxis sensitivity): [0.1, 10.0]]
</parameters>

Now, generate your PDE candidate:
"""
    return prompt


def create_critic_prompt(
    equation_str: str,
    metrics: Dict,
    iteration: int,
    include_comparison: bool = True
) -> str:
    """
    Create prompt for Visual Critic agent

    Args:
        equation_str: The PDE equation being evaluated
        metrics: Computed numerical metrics
        iteration: Current iteration number
        include_comparison: Whether to compare with previous best

    Returns:
        Formatted prompt string
    """
    prompt = f"""# Task: Visual Analysis of PDE Solution Quality

You are an expert in mathematical biology and scientific visualization. Your task is to
analyze visualizations of a PDE solution and provide detailed critique.

## Context

**Iteration:** {iteration}

**Proposed PDE:**
{equation_str}

**Numerical Metrics:**
- MSE (Mean Squared Error): {metrics.get('mse', 'N/A')}
- R² (Coefficient of Determination): {metrics.get('r2', 'N/A')}
- NMSE (Normalized MSE): {metrics.get('nmse', 'N/A')}
- Mass Conservation Error: {metrics.get('mass_error', 'N/A')}%

## Your Task

Analyze the visualization provided and evaluate the PDE solution quality across multiple dimensions:

### 1. Spatial Accuracy (0-10)
- Do predicted spatial patterns match observations?
- Are gradients and features captured correctly?
- Are there systematic spatial biases (e.g., center vs edges)?

### 2. Temporal Dynamics (0-10)
- Does the temporal evolution match observations?
- Are early, mid, and late dynamics captured?
- Does error grow or remain stable over time?

### 3. Physical Plausibility (0-10)
- Is mass approximately conserved?
- Are there non-physical artifacts (negative densities, discontinuities)?
- Do gradient fields make physical sense?
- Is the diffusion-chemotaxis balance reasonable?

### 4. Error Characteristics (0-10)
- Is error randomly distributed or systematic?
- What is the error magnitude relative to signal?
- Where (spatially/temporally) is error largest?

### 5. Overall Quality (0-10)
- Holistic assessment considering all factors
- Comparison to expected behavior
- Production readiness

## Output Format

Provide your analysis in the following structured format:

<scores>
Spatial Accuracy: [score 0-10]
Temporal Dynamics: [score 0-10]
Physical Plausibility: [score 0-10]
Error Characteristics: [score 0-10]
Overall Quality: [score 0-10]
Average Score: [average of above]
</scores>

<analysis>
[Detailed analysis paragraph covering:
- What the PDE does well
- What the PDE does poorly
- Specific observations from visualizations
- Physical interpretation of results
- Comparison to expected chemotaxis behavior]
</analysis>

<strengths>
[Bullet list of 2-3 key strengths]
</strengths>

<weaknesses>
[Bullet list of 2-3 key weaknesses]
</weaknesses>

<suggestions>
[Specific suggestions for improvement:
- Which terms to add/remove/modify?
- Parameter ranges to explore?
- Alternative functional forms?
- Physical considerations to incorporate?]
</suggestions>

<verdict>
[Overall verdict: "Excellent", "Good", "Acceptable", "Poor", or "Inadequate"]
[Brief justification for verdict]
</verdict>

Now, analyze the visualization and provide your detailed critique:
"""
    return prompt


def create_initial_hypothesis_prompt(data_summary: Dict) -> str:
    """
    Create prompt for initial hypothesis generation before any iterations

    Args:
        data_summary: Statistical summary of data

    Returns:
        Formatted prompt for initial hypotheses
    """
    prompt = f"""# Task: Generate Initial PDE Hypotheses for Chemotaxis

Based on the data summary below, generate 3-5 initial PDE hypotheses to explore.

{CHEMOTAXIS_CONTEXT}

{OPERATOR_LIBRARY}

## Data Summary
- Shape: {data_summary.get('shape', 'N/A')}
- Cell density (g) range: [{data_summary.get('g_min', 0):.4f}, {data_summary.get('g_max', 1):.4f}]
- Mean density: {data_summary.get('g_mean', 'N/A'):.4f}
- Total mass change: {data_summary.get('mass_change_pct', 'N/A'):.2f}%

## Task

Generate 3-5 diverse PDE hypotheses ranging from simple to complex:

1. **Simple**: Pure diffusion or simple chemotaxis
2. **Intermediate**: Standard Keller-Segel or variants
3. **Complex**: Nonlinear terms or additional phenomena

For each hypothesis, provide:
- Equation
- Physical interpretation
- Expected behavior
- Parameter ranges

Format each as:

### Hypothesis [N]: [Name]

**Equation:** [PDE using ∇, ∇·, Δ, ∂/∂t]

**Interpretation:** [What does each term represent?]

**Expected Behavior:** [What patterns should emerge?]

**Parameters:** [List with ranges]

Now generate your initial hypotheses:
"""
    return prompt


def extract_equation_from_response(response: str) -> str:
    """
    Extract equation from structured response

    Args:
        response: Full response text

    Returns:
        Extracted equation string
    """
    import re

    # Try to find content between <equation> tags
    match = re.search(r'<equation>(.*?)</equation>', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: look for lines that look like equations
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        # Check if line contains PDE operators
        if any(op in line for op in ['∇', 'Δ', '∂', 'grad', 'div', 'laplacian']):
            return line

    return ""


def extract_reasoning_from_response(response: str) -> str:
    """Extract reasoning from structured response"""
    import re

    match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_scores_from_critique(response: str) -> Dict[str, float]:
    """
    Extract numerical scores from Visual Critic response

    Args:
        response: Critic's full response

    Returns:
        Dictionary of scores
    """
    import re

    scores = {}

    # Try to find scores section
    match = re.search(r'<scores>(.*?)</scores>', response, re.DOTALL)
    if not match:
        return scores

    scores_text = match.group(1)

    # Extract individual scores
    patterns = {
        'spatial': r'Spatial Accuracy:\s*(\d+(?:\.\d+)?)',
        'temporal': r'Temporal Dynamics:\s*(\d+(?:\.\d+)?)',
        'physical': r'Physical Plausibility:\s*(\d+(?:\.\d+)?)',
        'error': r'Error Characteristics:\s*(\d+(?:\.\d+)?)',
        'overall': r'Overall Quality:\s*(\d+(?:\.\d+)?)',
        'average': r'Average Score:\s*(\d+(?:\.\d+)?)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, scores_text, re.IGNORECASE)
        if match:
            scores[key] = float(match.group(1))

    return scores


def extract_suggestions_from_critique(response: str) -> str:
    """Extract suggestions from Visual Critic response"""
    import re

    match = re.search(r'<suggestions>(.*?)</suggestions>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_verdict_from_critique(response: str) -> str:
    """Extract verdict from Visual Critic response"""
    import re

    match = re.search(r'<verdict>(.*?)</verdict>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""
