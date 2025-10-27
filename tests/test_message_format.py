#!/usr/bin/env python3
"""
Quick test to verify the AsyncLLMClient fix works
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing AsyncLLMClient message format fix...")

# Test that UserMessage import works
try:
    from autogen_core.components.models import UserMessage
    print("✓ UserMessage import successful")

    # Test creating a UserMessage
    msg = UserMessage(content="Test prompt", source="user")
    print(f"✓ UserMessage created: {msg}")

except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nTrying alternative import...")
    try:
        from autogen_agentchat.messages import UserMessage
        print("✓ Alternative UserMessage import successful")
        msg = UserMessage(content="Test prompt", source="user")
        print(f"✓ UserMessage created: {msg}")
    except ImportError as e2:
        print(f"✗ Alternative import also failed: {e2}")

print("\nTest complete!")
