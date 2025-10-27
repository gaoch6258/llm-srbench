# Quick reference for fixes needed:

## Issue 1: Agent should generate CODE directly, not natural language
- ✅ FIXED: Changed tool signature from `pde_description` to `pde_code`
- ✅ FIXED: Tool now accepts complete Python code
- ✅ FIXED: Removed intermediate LLM call in tool (agent IS the LLM)
- ⚠️ TODO: Update agent system_message to tell it to generate CODE
- ⚠️ TODO: Update SUPPORTED_FORMS_GUIDE with code examples and scipy usage

## Issue 2: Missing multi-agent architecture
- ❌ TODO: Add Manager agent to coordinate
- ❌ TODO: Add Critic agent for visual analysis
- ❌ TODO: Add Worker agents to generate diverse hypotheses
- ❌ TODO: Use RoundRobinGroupChat or similar

## Key Changes to Make:

### 1. Update SUPPORTED_FORMS_GUIDE (lines 42-120)
Add:
- Explicit scipy.ndimage usage (laplace, sobel, etc.)
- Complete code examples (not just templates)
- Emphasize: "Generate COMPLETE executable code"

### 2. Update agent system_message (lines 582-611)
Change from:
```
For each hypothesis, call evaluate_pde_tool(pde_description="...", num_params=N)
```

To:
```
For each hypothesis:
1. Write COMPLETE Python code for pde_update function
2. Call evaluate_pde_tool(pde_code="<your complete code>", num_params=N)
```

### 3. Add Multi-Agent Architecture (NEW)
Replace single AssistantAgent with:
```python
# Create agents
manager = AssistantAgent(name="Manager", ...)
critic = AssistantAgent(name="Critic", ...)  # For visual analysis
worker1 = AssistantAgent(name="Worker1", ...)  # Generates code
worker2 = AssistantAgent(name="Worker2", ...)  # Generates code

# Create team
from autogen_agentchat.teams import RoundRobinGroupChat
team = RoundRobinGroupChat([manager, critic, worker1, worker2])
```

## Status:
- [x] Tool signature changed to pde_code
- [x] Tool implementation updated (no intermediate LLM)
- [ ] SUPPORTED_FORMS_GUIDE needs scipy examples
- [ ] Agent prompts need "generate CODE" instructions
- [ ] Multi-agent architecture not yet implemented

## Next Steps:
1. Update lines 42-120: SUPPORTED_FORMS_GUIDE with scipy code examples
2. Update lines 582-611: Agent system message to emphasize CODE generation
3. Add multi-agent setup around line 574
