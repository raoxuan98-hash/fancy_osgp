"""
Backwards compatibility module for SubspaceLoRA CLIP learner.

This module imports the refactored components to maintain compatibility
with existing code that imports from this module.
"""

# Import the main class from the refactored module
from models.subspace_lora_clip_learner import SubspaceLoRAClipLearner

# Re-export for backwards compatibility
__all__ = ['SubspaceLoRAClipLearner', 'SubspaceLoRA_CLIP']

# Create the backwards compatibility alias
SubspaceLoRA_CLIP = SubspaceLoRAClipLearner
