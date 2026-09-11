"""PII-safe regex and Unicode text cleaning.

The default cleaner deliberately keeps casing, digits, and punctuation used
inside common PII values. More destructive linguistic transformations live in
``linguistic.py`` and are opt-in.
"""

from __future__ import annotations

import html
import re
import unicodedata


_HTML_TAG_RE = re.compile(r"<[^>]*>")
_WHITESPACE_RE = re.compile(r"\s+")

# Broad emoji/pictograph ranges. Bengali and other ordinary scripts are not
# included, so multilingual text survives cleaning.
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # flags
    "\U0001F300-\U0001F5FF"  # symbols and pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport and map symbols
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"          # miscellaneous symbols
    "\u2700-\u27BF"          # dingbats
    "]+",
    flags=re.UNICODE,
)
_VARIATION_AND_JOINER_RE = re.compile(r"[\u200d\ufe0e\ufe0f]")


def clean_text(text: str, *, remove_emoji: bool = True) -> str:
    """Return minimally normalized text without destroying PII signals.

    The function applies NFKC normalization, strips HTML tags using a regular
    expression, decodes HTML entities, optionally removes emoji, and collapses
    whitespace. It intentionally preserves case, digits, ``@``, ``.``, ``-``,
    ``+``, and all other non-HTML punctuation.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFKC", text)
    without_tags = _HTML_TAG_RE.sub(" ", normalized)
    decoded = html.unescape(without_tags)
    if remove_emoji:
        decoded = _EMOJI_RE.sub(" ", decoded)
        decoded = _VARIATION_AND_JOINER_RE.sub("", decoded)
    return _WHITESPACE_RE.sub(" ", decoded).strip()
