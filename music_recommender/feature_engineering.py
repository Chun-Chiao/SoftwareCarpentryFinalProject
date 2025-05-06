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
          - List of feature names (after weighting and PCA)
    """
    # Copy metadata and track order
    meta = metadata.copy()
    track_ids = meta['track_id'].tolist()

    # Encode genre as one-hot
    genre_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    genre_array = genre_encoder.fit_transform(meta[['track_genre']])
    raw_genre_cols = genre_encoder.get_feature_names_out(['track_genre'])
    genre_cols = [col.replace('track_genre_', 'genre_') for col in raw_genre_cols]
    # Apply genre weight
    genre_array *= genre_weight

    # Numeric features
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
    numeric_scaled *= audio_weight

    # Combine features
    features = np.hstack([genre_array, numeric_scaled])
    feature_names = genre_cols + numeric_cols

    # PCA reduction
    if pca_components is not None and pca_components > 0:
        pca = PCA(n_components=pca_components)
        features = pca.fit_transform(features)
        feature_names = [f"pc_{i+1}" for i in range(pca_components)]

    # Convert to sparse
    mat = csr_matrix(features)
    return mat, track_ids, feature_names


def compute_content_similarity(
    seed_id: str,
    content_mat: csr_matrix,
    track_ids: List[str]
) -> np.ndarray:
    """
    Compute cosine similarity vector between seed track and all tracks,
    excluding the seed by setting its similarity to -inf.
    """
    if seed_id not in track_ids:
        raise ValueError(f"Seed track ID '{seed_id}' not found")
    idx = track_ids.index(seed_id)
    dense = content_mat.toarray()
    seed_vec = dense[idx]
    norms = np.linalg.norm(dense, axis=1) * np.linalg.norm(seed_vec)
    norms[norms == 0] = 1e-10
    sims = (dense @ seed_vec) / norms
    # Exclude self
    sims[idx] = -np.inf
    return sims


def hybrid_scores(
    content_sims: np.ndarray,
    popularity: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Combine content similarity and popularity into a hybrid score.
    """
    pop_norm = (popularity - popularity.min()) / (popularity.max() - popularity.min() + 1e-10)
    return alpha * content_sims + (1 - alpha) * pop_norm


def recommend_hybrid(
    seed_id: str,
    content_mat=None,
    track_ids=None,
    metadata=None,
    genre_weight: float = 1.0,
    audio_weight: float = 1.0,
    pca_components: int = None,
    alpha: float = 0.5,
    top_n: int = 10
) -> List[str]:
    """
    Recommend top_n tracks using hybrid content+popularity scores,
    excluding the seed itself and limiting recommendations to available tracks.
    """
    if metadata is None:
        raise ValueError("`metadata` must be provided when `content_mat` is None")

    # Rebuild feature matrix with weights/PCA
    mat, ids, _ = build_content_matrix(
        metadata,
        genre_weight=genre_weight,
        audio_weight=audio_weight,
        pca_components=pca_components
    )

    # Compute content similarity scores (seed excluded)
    sims = compute_content_similarity(seed_id, mat, ids)

    # Get popularity array
    pops = metadata['popularity'].to_numpy(dtype=float)

    # Compute hybrid scores
    hybrid = hybrid_scores(sims, pops, alpha=alpha)

    # Rank tracks by descending hybrid score
    ranked_idxs = np.argsort(hybrid)[::-1]

    # Determine how many to return (exclude seed)
    max_recs = max(0, len(ids) - 1)
    k = min(top_n, max_recs)

    # Select top k recommendations
    top_idxs = ranked_idxs[:k]
    return [ids[i] for i in top_idxs]
