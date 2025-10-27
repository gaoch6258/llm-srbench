#!/usr/bin/env python3
"""
Test AsyncLLMClient in isolation
"""
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily
from autogen_agentchat.messages import UserMessage
from autogen_core import CancellationToken

# AsyncLLMClient from run_pde_discovery_autogen_v04.py
class AsyncLLMClient:
    """Async-compatible LLM client wrapper for LLMSR"""
    def __init__(self, model_client):
        self.model_client = model_client

    def generate(self, prompt):
        """Synchronous wrapper that handles async properly"""
        import asyncio
        from autogen_agentchat.messages import UserMessage

        async def _async_generate():
            from autogen_core import CancellationToken
            # Use proper message format for AutoGen v0.4
            response = await self.model_client.create(
                messages=[UserMessage(content=prompt, source="user")],
                cancellation_token=CancellationToken()
            )
            return response.content

        # Create new loop for this thread
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            result = new_loop.run_until_complete(_async_generate())
            return result
        finally:
            new_loop.close()


async def main():
    print("Testing AsyncLLMClient...")

    # Create model client
    model_client = OpenAIChatCompletionClient(
        model="/mnt/hdd_raid5/gaoch/Qwen3-VL-8B-Instruct",
        base_url="http://localhost:10005/v1",
        api_key="EMPTY",
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
    )
    print("✓ Model client created")

    # Test direct API call (async)
    print("\n1. Testing direct async API call...")
    response = await model_client.create(
        messages=[UserMessage(content="Say 'Hello from direct call'", source="user")],
        cancellation_token=CancellationToken()
    )
    print(f"   Response: {response.content[:100]}")

    # Test AsyncLLMClient wrapper (sync)
    print("\n2. Testing AsyncLLMClient wrapper...")
    llm_client = AsyncLLMClient(model_client)
    result = llm_client.generate("Say 'Hello from AsyncLLMClient'")
    print(f"   Result: {result[:100]}")

    # Close model client
    await model_client.close()
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
