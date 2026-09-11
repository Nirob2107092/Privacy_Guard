"""Small-vocabulary, PII-aware spelling correction."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence


DOMAIN_VOCABULARY: tuple[str, ...] = (
    "account",
    "address",
    "amount",
    "bank",
    "call",
    "confirm",
    "contact",
    "email",
    "name",
    "number",
    "payment",
    "phone",
    "please",
    "send",
    "transfer",
)

# Common informal forms whose intended expansion is unambiguous but can be
# farther away than a conservative edit-distance threshold (for example acc).
DOMAIN_ALIASES: dict[str, str] = {
    "acc": "account",
    "acct": "account",
    "addr": "address",
    "amt": "amount",
    "phn": "phone",
    "plz": "please",
    "pymnt": "payment",
}


def levenshtein_distance(source: str, target: str) -> int:
    """Compute Levenshtein edit distance using dynamic programming."""

    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    # Keep only one DP row, reducing memory from O(m*n) to O(min(m, n)).
    if len(source) < len(target):
        source, target = target, source
    previous = list(range(len(target) + 1))
    for source_index, source_char in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_char in enumerate(target, start=1):
            insertion = current[target_index - 1] + 1
            deletion = previous[target_index] + 1
            substitution = previous[target_index - 1] + (source_char != target_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _restore_case(original: str, corrected: str) -> str:
    if original.isupper():
        return corrected.upper()
    if original.istitle():
        return corrected.title()
    return corrected


def correct_token(
    token: str,
    *,
    vocabulary: Iterable[str] = DOMAIN_VOCABULARY,
    max_distance: int = 2,
) -> str:
    """Correct one informal context token while protecting possible PII.

    Tokens containing any digit are always returned untouched. Tokens with
    punctuation (such as emails, URLs, and signed identifiers) are also left
    alone; correction is restricted to alphabetic context words.
    """

    if any(char.isdigit() for char in token) or not re.fullmatch(r"[A-Za-z]+", token):
        return token

    lowered = token.lower()
    if lowered in DOMAIN_ALIASES:
        return _restore_case(token, DOMAIN_ALIASES[lowered])

    candidates = tuple(dict.fromkeys(word.lower() for word in vocabulary))
    if not candidates:
        return token

    best = min(candidates, key=lambda word: (levenshtein_distance(lowered, word), word))
    distance = levenshtein_distance(lowered, best)
    if distance > max_distance:
        return token
    return _restore_case(token, best)


def correct_spelling(
    tokens: Sequence[str],
    *,
    vocabulary: Iterable[str] = DOMAIN_VOCABULARY,
    max_distance: int = 2,
) -> list[str]:
    """Correct a token sequence against the small domain vocabulary."""

    return [
        correct_token(token, vocabulary=vocabulary, max_distance=max_distance)
        for token in tokens
    ]
