"""Optional Gensim pretrained embedding adapter."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def load_pretrained_embeddings(
    vocabulary: Iterable[str],
    *,
    model_name: str = "glove-wiki-gigaword-100",
) -> tuple[dict[str, int], np.ndarray] | None:
    """Load Gensim vectors and adapt them to the project's vocabulary.

    Gensim and its model download are deliberately optional. Any import,
    network, or model lookup failure returns ``None`` so local experiments
    still complete with the scratch-trained arm.
    """

    try:
        import gensim.downloader as api

        model = api.load(model_name)
        words = sorted({word.lower() for word in vocabulary})
        known_words = [word for word in words if word in model]
        if not known_words:
            raise ValueError("the pretrained model contains none of the project words")
        word2id = {word: index for index, word in enumerate(known_words)}
        matrix = np.asarray([model[word] for word in known_words], dtype=np.float64)
        print(f"Loaded pretrained {model_name}: {matrix.shape}")
        return word2id, matrix
    except Exception as error:
        print(
            "pretrained embeddings unavailable, skipping this comparison arm "
            f"({type(error).__name__}: {error})"
        )
        return None
