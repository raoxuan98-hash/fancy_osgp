"""Prompt templates shared between CLIP and ViT style training."""
from __future__ import annotations

from typing import Callable, List


PromptTemplate = Callable[[str], str]


def build_cifar_template() -> List[PromptTemplate]:
    """Standard prompt engineering recipe used by CLIP on CIFAR style datasets."""

    return [
        lambda c: f"a photo of a {c}.",
        lambda c: f"a blurry photo of a {c}.",
        lambda c: f"a black and white photo of a {c}.",
        lambda c: f"a low contrast photo of a {c}.",
        lambda c: f"a high contrast photo of a {c}.",
        lambda c: f"a bad photo of a {c}.",
        lambda c: f"a good photo of a {c}.",
        lambda c: f"a photo of a small {c}.",
        lambda c: f"a photo of a big {c}.",
        lambda c: f"a photo of the {c}.",
        lambda c: f"a blurry photo of the {c}.",
        lambda c: f"a black and white photo of the {c}.",
        lambda c: f"a low contrast photo of the {c}.",
        lambda c: f"a high contrast photo of the {c}.",
        lambda c: f"a bad photo of the {c}.",
        lambda c: f"a good photo of the {c}.",
        lambda c: f"a photo of the small {c}.",
        lambda c: f"a photo of the big {c}.",
    ]


def build_default_template() -> List[PromptTemplate]:
    """Fallback prompt template for datasets without specialised wording."""

    return [
        lambda c: f"a photo of a {c}.",
        lambda c: f"a photo of the {c}.",
    ]
