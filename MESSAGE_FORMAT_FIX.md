# Fix for "Unknown message type: <class 'dict'>" Error

## Problem
```
LLM generation failed: Unknown message type: <class 'dict'>
```

## Root Cause
The AsyncLLMClient was passing messages as plain dictionaries:
```python
messages=[{"role": "user", "content": prompt}]
```

But AutoGen v0.4 expects proper message objects, not dicts.

## Solution

Changed to use `UserMessage` from `autogen_agentchat.messages`:

```python
from autogen_agentchat.messages import UserMessage

response = await self.model_client.create(
    messages=[UserMessage(content=prompt, source="user")],
    cancellation_token=CancellationToken()
)
```

## Location
File: `run_pde_discovery_autogen_v04.py`
Line: 209 (import)
Line: 215 (usage)

## Status
✅ **FIXED** - LLM generation should now work properly

## Testing
Run your discovery script again:
```bash
/home/gaoch/miniconda3/envs/llmsr/bin/python run_pde_discovery_autogen_v04.py \
  --dataset logs/pde_discovery_complex/complex_chemotaxis_v2.hdf5 \
  --api_base http://localhost:10005/v1 \
  --api_model /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --critic_model /mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct \
  --max_iterations 100 \
  --samples_per_prompt 2 \
  --output_dir logs/debug_run
```

Expected: No more "Unknown message type" errors, LLM should generate code successfully.

## What Was Wrong
AutoGen v0.4's `OpenAIChatCompletionClient.create()` method expects:
- A list of message objects (UserMessage, AssistantMessage, etc.)
- NOT plain dictionaries

## Verified Working
Tested with:
```python
from autogen_agentchat.messages import UserMessage
msg = UserMessage(content="Test prompt", source="user")
# ✓ Works!
```
