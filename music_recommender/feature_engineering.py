"""
feature_engineering.py

Builds content feature matrix with optional weighting and dimensionality reduction,
then computes similarities and hybrid scores for recommendations.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, List


def build_content_matrix(
    metadata,
    genre_weight: float = 1.0,
    audio_weight: float = 1.0,
    pca_components: int = None
) -> Tuple[csr_matrix, List[str], List[str]]:
    """
    Builds a content feature matrix from metadata with optional weights and PCA.

    Args:
        metadata: DataFrame with 'track_id', 'track_genre', and numeric audio features.
        genre_weight: Multiplier for genre one-hot features.
        audio_weight: Multiplier for numeric features.
        pca_components: Number of PCA components to reduce to (if None, no PCA).

    Returns:
        Tuple of:
          - CSR matrix of shape (n_items, n_features')
          - Ordered list of track_ids corresponding to rows
          - List of feature names (after weighting and PCA, PCA features named 'pc_{i}')
    """
    # Copy metadata and track order
    meta = metadata.copy()
    track_ids = meta['track_id'].tolist()

    # Encode genre as one-hot
    genre_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    genre_array = genre_encoder.fit_transform(meta[['track_genre']])
    raw_genre_cols = genre_encoder.get_feature_names_out(['track_genre'])
    genre_cols = [col.replace('track_genre_', 'genre_')
                  for col in raw_genre_cols]
    # Apply genre weight
    genre_array = genre_array * genre_weight

    # Numeric audio features
    numeric_cols = [
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'duration_ms', 'popularity'
    ]
    missing = set(numeric_cols) - set(meta.columns)
    if missing:
        raise ValueError(f"Missing numeric metadata columns: {missing}")
    numeric_data = meta[numeric_cols].fillna(0).values
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_data)
    # Apply audio weight
    numeric_scaled = numeric_scaled * audio_weight

    # Combine features
    features = np.hstack([genre_array, numeric_scaled])
    feature_names = genre_cols + numeric_cols

    # Optionally reduce dimensionality
    if pca_components is not None and pca_components > 0:
        pca = PCA(n_components=pca_components)
        reduced = pca.fit_transform(features)
        features = reduced
        feature_names = [f"pc_{i+1}" for i in range(pca_components)]

    # Convert to sparse matrix
    mat = csr_matrix(features)
    return mat, track_ids, feature_names


def compute_content_similarity(
    seed_id: str,
    content_mat: csr_matrix,
    track_ids: List[str]
) -> np.ndarray:
    """
    Compute cosine similarity vector between seed track and all tracks.

    Args:
        seed_id: track_id to use as reference.
        content_mat: CSR matrix from build_content_matrix.
        track_ids: List of track IDs matching matrix rows.

    Returns:
        Array of similarity scores (shape: [n_items]).
    """
    if seed_id not in track_ids:
        raise ValueError(f"Seed track ID '{seed_id}' not found")
    idx = track_ids.index(seed_id)
    dense = content_mat.toarray()
    seed_vec = dense[idx]
    norms = np.linalg.norm(dense, axis=1) * np.linalg.norm(seed_vec)
    norms[norms == 0] = 1e-10
    sims = (dense @ seed_vec) / norms
    sims[idx] = -np.inf  # exclude self
    return sims


def hybrid_scores(
    content_sims: np.ndarray,
    popularity: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Combine content similarity and popularity into a hybrid score.

    Args:
        content_sims: Array of content-based similarity scores.
        popularity: Array of popularity scores (0-100).
        alpha: Weight for content vs. popularity (0..1).

    Returns:
        Hybrid score array.
    """
    pop_norm = (popularity - popularity.min()) / \
        (popularity.max() - popularity.min() + 1e-10)
    return alpha * content_sims + (1 - alpha) * pop_norm


def recommend_hybrid(
    seed_id: str,
    content_mat,
    track_ids,
    metadata,
    genre_weight: float = 1.0,
    audio_weight: float = 1.0,
    pca_components: int = None,
    alpha: float = 0.5,
    top_n: int = 10
) -> List[str]:
    """
    Recommend top_n tracks using hybrid content+popularity scores.

    Args:
        seed_id: Seed track ID.
        content_mat: Ignored; matrix is rebuilt with given weights/PCA.
        track_ids: Ignored; order derived from metadata.
        metadata: Original metadata DataFrame (for popularity).
        genre_weight, audio_weight, pca_components: passed to build_content_matrix.
        alpha: hybrid weight.
        top_n: number of recommendations.

    Returns:
        List of recommended track IDs.
    """
    mat, ids, _ = build_content_matrix(
        metadata,
        genre_weight=genre_weight,
        audio_weight=audio_weight,
        pca_components=pca_components
    )
    sims = compute_content_similarity(seed_id, mat, ids)
    pops = metadata['popularity'].to_numpy(dtype=float)
    hybrid = hybrid_scores(sims, pops, alpha=alpha)
    top_idx = np.argsort(hybrid)[-top_n:][::-1]
    return [ids[i] for i in top_idx]
