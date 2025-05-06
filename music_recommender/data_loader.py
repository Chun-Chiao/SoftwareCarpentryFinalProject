"""
data_loader.py

Loads interactions and track metadata from local CSV files.
Removes duplicate track rows based on identical artist, album, and track name.
"""

import pandas as pd
from typing import Tuple

def load_interactions(path: str) -> pd.DataFrame:
    """
    Load user–item interactions from a CSV file.

    Expects columns: ['user_id', 'track_id', 'rating'], in that exact order.
    """
    df = pd.read_csv(path)
    expected_cols = ['user_id', 'track_id', 'rating']
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in interactions file: {missing}")
    # Enforce the exact column ordering
    return df[expected_cols].copy()


def load_metadata(path: str) -> pd.DataFrame:
    """
    Load track metadata from a CSV file containing Spotify features.
    Removes duplicates where artists, album_name, and track_name all match.

    Expects columns:
      - track_id
      - artists
      - album_name
      - track_name
      - popularity
      - duration_ms
      - explicit
      - danceability
      - energy
      - key
      - loudness
      - mode
      - speechiness
      - acousticness
      - instrumentalness
      - liveness
      - valence
      - tempo
      - time_signature
      - track_genre

    Args:
        path: Path to the metadata CSV.

    Returns:
        DataFrame with one row per unique track_id after duplicate removal.
    """
    df = pd.read_csv(path)

    # Drop unnamed index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Remove duplicate rows based on artists, album_name, and track_name
    dedup_cols = ['artists', 'track_name']
    df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    # Verify expected columns
    expected_cols = [
        'track_id', 'artists', 'album_name', 'track_name',
        'popularity', 'duration_ms', 'explicit',
        'danceability', 'energy', 'key', 'loudness',
        'mode', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence',
        'tempo', 'time_signature', 'track_genre'
    ]
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    # Ensure each track_id is unique (keep first occurrence of any duplicates)
    metadata = df.drop_duplicates(subset=['track_id']).reset_index(drop=True)
    return metadata.copy()
