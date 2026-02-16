"""
Semantic schema discovery for Excel using Sentence-Transformers (all-mpnet-base-v2).
Embeds system field descriptions once; matches Excel labels by cosine similarity.
Schema-agnostic: works for both row-oriented and column-oriented layouts.
"""
import logging
from typing import List, Tuple

import numpy as np

from utils.system_field_mapping import SYSTEM_FIELD_MAPPING

logger = logging.getLogger(__name__)

# Lazy-loaded model and cached description embeddings
_model = None
_desc_embeddings = None
_description_keys: List[str] = []
_description_texts: List[str] = []


def _get_model():
    """Load Sentence-Transformer model once (lazy)."""
    global _model
    if _model is None:
        # Reduce Hugging Face Hub log noise (unauthenticated-request warning)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            msg = (
                "sentence_transformers is not installed. Install it with: "
                "pip install sentence-transformers"
            )
            logger.error(f"Failed to load SentenceTransformer: {e}. {msg}")
            raise ImportError(msg) from e
        try:
            _model = SentenceTransformer("all-mpnet-base-v2")
            logger.info("Loaded SentenceTransformer all-mpnet-base-v2")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            raise
    return _model


def _ensure_description_embeddings() -> Tuple[np.ndarray, List[str], List[str]]:
    """Build and cache description embeddings. Returns (embeddings, keys, texts)."""
    global _desc_embeddings, _description_keys, _description_texts
    if _desc_embeddings is not None:
        return _desc_embeddings, _description_keys, _description_texts

    # Stable order: list of (system_field_key, description)
    keys = []
    texts = []
    for field_key, (_scale_base, description) in SYSTEM_FIELD_MAPPING.items():
        keys.append(field_key)
        texts.append(description)
    _description_keys = keys
    _description_texts = texts

    model = _get_model()
    _desc_embeddings = model.encode(
        _description_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    logger.debug(f"Cached embeddings for {len(_description_texts)} system field descriptions")
    return _desc_embeddings, _description_keys, _description_texts


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Embed a list of strings; returns (n, 768) normalized embeddings."""
    if not texts:
        return np.array([]).reshape(0, 768)
    model = _get_model()
    arr = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.atleast_2d(arr)


def match_labels_to_system_fields(
    label_texts: List[str],
    similarity_threshold: float = 0.45,
) -> List[Tuple[int, str, float]]:
    """
    For each label text, find the best-matching system field by cosine similarity.

    Args:
        label_texts: List of strings (e.g. Excel header labels).
        similarity_threshold: Minimum cosine similarity to count as a match.

    Returns:
        List of (label_index, system_field_key, score) for labels that match above threshold.
        Unmatched labels are omitted.
    """
    if not label_texts:
        return []

    desc_emb, keys, _ = _ensure_description_embeddings()
    unit_emb = get_embeddings(label_texts)

    # (n_labels, n_descriptions)
    scores = np.dot(unit_emb, desc_emb.T)

    results = []
    for i in range(len(label_texts)):
        best_idx = int(scores[i].argmax())
        best_score = float(scores[i][best_idx])
        if best_score >= similarity_threshold:
            results.append((i, keys[best_idx], best_score))
    return results


def get_system_field_keys_and_descriptions() -> List[Tuple[str, str]]:
    """Return [(system_field_key, description), ...] in stable order (for external use)."""
    _ensure_description_embeddings()
    return list(zip(_description_keys, _description_texts))
