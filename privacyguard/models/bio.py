"""Utilities for reconstructing valid BIO sequences from token categories."""

from __future__ import annotations

from collections.abc import Sequence


def categories_to_bio(categories: Sequence[str], *, outside_label: str = "O") -> list[str]:
    """Convert category predictions to BIO tags from left to right.

    Adjacent tokens with the same non-outside category form one span. A change
    of category, or an outside token, starts a new span.
    """

    tags: list[str] = []
    previous = outside_label
    for category in categories:
        if category == outside_label:
            tags.append(outside_label)
        else:
            prefix = "I" if category == previous else "B"
            tags.append(f"{prefix}-{category}")
        previous = category
    return tags
