"""Synthetic BIO-tagged PII dataset generation."""

from .generator import generate_examples, generate_group_example, split_examples

__all__ = ["generate_examples", "generate_group_example", "split_examples"]
