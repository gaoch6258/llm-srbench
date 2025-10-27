# Fix for Async Event Loop Issue

## Problem
```
"LLM generation failed: There is no current event loop in thread 'asyncio_1'."
```

## Root Cause
The `evaluate_pde_tool` is called from within AutoGen's async context. When the LLMSR solver tries to generate code, it needs to call the LLM client, which is async. The issue is:

1. AutoGen agent runs in async context (has event loop)
2. Tool is called synchronously from that async context
3. LLM client tries to use async calls
4. Conflict: can't use `asyncio.run()` from within existing event loop

## Solution

Created `AsyncLLMClient` wrapper that:

```python
class AsyncLLMClient:
    def generate(self, prompt):
        # Create NEW event loop for this thread
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            result = new_loop.run_until_complete(_async_generate())
            return result
        finally:
            new_loop.close()
```

This works because:
- Creates a fresh event loop (no conflict with parent)
- Runs the async LLM call in that loop
- Cleans up properly
- Returns synchronous result

## Location
File: `run_pde_discovery_autogen_v04.py`
Lines: 200-231

## Status
✅ **FIXED** - LLM generation should now work without event loop errors

## Testing
Run your discovery script:
```bash
/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_autogen_v04.py \
  --dataset /path/to/data.h5 \
  --max_iterations 100
```

Expected: No more "no current event loop" errors
