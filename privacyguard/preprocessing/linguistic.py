"""NLTK wrappers with destructive transformations disabled by default."""

from __future__ import annotations

from collections.abc import Sequence

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from .cleaning import clean_text


def tokenize(text: str) -> list[str]:
    """Tokenize text with NLTK's ``word_tokenize``."""

    return word_tokenize(text)


def transform_tokens(
    tokens: Sequence[str],
    *,
    lowercase: bool = False,
    remove_stopwords: bool = False,
    stem: bool = False,
    lemmatize: bool = False,
) -> list[str]:
    """Apply optional Lab 1 linguistic transformations to tokens.

    Punctuation and structured PII tokens are retained. Stop words are compared
    case-insensitively even when output lowercasing is disabled.
    """

    result = list(tokens)
    if lowercase:
        result = [token.lower() for token in result]

    if remove_stopwords:
        english_stopwords = set(stopwords.words("english"))
        result = [token for token in result if token.lower() not in english_stopwords]

    if stem:
        stemmer = PorterStemmer()
        result = [stemmer.stem(token) if token.isalpha() else token for token in result]

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        result = [
            lemmatizer.lemmatize(token) if token.isalpha() else token
            for token in result
        ]

    return result


def preprocess_text(
    text: str,
    *,
    remove_emoji: bool = True,
    lowercase: bool = False,
    remove_stopwords: bool = False,
    stem: bool = False,
    lemmatize: bool = False,
) -> list[str]:
    """Clean, tokenize, and optionally transform text.

    All signal-destructive flags default to ``False`` for PII safety.
    """

    cleaned = clean_text(text, remove_emoji=remove_emoji)
    tokens = tokenize(cleaned)
    return transform_tokens(
        tokens,
        lowercase=lowercase,
        remove_stopwords=remove_stopwords,
        stem=stem,
        lemmatize=lemmatize,
    )
