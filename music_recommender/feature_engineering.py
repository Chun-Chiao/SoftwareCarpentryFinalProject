"""
feature_engineering.py

Builds content feature matrix for recommendation when no user interactions are available.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_content_matrix(metadata: pd.DataFrame) -> csr_matrix:
    """
    Builds a content feature matrix from metadata DataFrame.

    Args:
        metadata: DataFrame with columns including 'track_id', 'track_genre',
                  and numeric audio features (danceability, energy, etc.).

    Returns:
        tuple: (csr_matrix of shape (n_items, n_features),
                list of track_ids in input order,
                list of feature names in column order)
    """
    # Copy metadata and extract track IDs
    meta = metadata.copy()
    track_ids = meta['track_id'].tolist()

    # One-hot encode genre
    genre_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    genre_array = genre_encoder.fit_transform(meta[['track_genre']])
    raw_genre_cols = genre_encoder.get_feature_names_out()
    genre_cols = [col.replace('track_genre_', 'genre_') for col in raw_genre_cols]

    # Numeric features
    numeric_cols = [
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'duration_ms', 'popularity'
    ]
    missing_nums = set(numeric_cols) - set(meta.columns)
    if missing_nums:
        raise ValueError(f"Missing numeric metadata columns: {missing_nums}")
    numeric_data = meta[numeric_cols].fillna(0).values

    # Standardize numeric features
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_data)

    # Combine genre and numeric features
    features = np.hstack([genre_array, numeric_scaled])
    feature_names = genre_cols + numeric_cols

    # Convert to sparse matrix for efficiency
    mat = csr_matrix(features)
    return mat, track_ids, feature_names


def recommend_similar_tracks(track_id: str,
                             content_mat: csr_matrix,
                             track_ids: list,
                             feature_names: list,
                             top_n: int = 10) -> list:
    """
    Recommend top_n tracks most similar to a given track based on content features.

    Args:
        track_id: The seed track identifier.
        content_mat: CSR matrix from build_content_matrix.
        track_ids: Ordered list of track IDs corresponding to matrix rows.
        feature_names: List of feature names (not used here but returned for reference).
        top_n: Number of similar tracks to return.

    Returns:
        List of track IDs most similar to the seed track (excluding itself).
    """
    if track_id not in track_ids:
        raise ValueError(f"Track ID '{track_id}' not found in metadata")
    idx = track_ids.index(track_id)
    # Compute cosine similarity
    row = content_mat[idx].toarray()
    # Normalize rows
    norms = np.linalg.norm(content_mat.toarray(), axis=1, keepdims=True)
    sim = (content_mat.toarray() @ row.T).flatten() / (norms.flatten() * np.linalg.norm(row))
    # Exclude self
    sim[idx] = -1
    # Get top indices
    top_idx = np.argsort(sim)[-top_n:][::-1]
    return [track_ids[i] for i in top_idx]
