"""
data_loader.py

Loads interactions and track metadata from local CSV files.
"""

import pandas as pd
from typing import Tuple


def load_interactions(path: str) -> pd.DataFrame:
    """
    Load user–item interactions from a CSV file.

    Expects columns: ['user_id', 'track_id', 'rating'].

    Args:
        path: Path to the interactions CSV.

    Returns:
        DataFrame with ['user_id', 'track_id', 'rating'].
    """
    df = pd.read_csv(path)
    expected = {'user_id', 'track_id', 'rating'}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in interactions file: {missing}")
    return df[list(expected)].copy()


def load_metadata(path: str) -> pd.DataFrame:
    """
    Load track metadata from a CSV file containing Spotify features.

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
        DataFrame with one row per track_id and all feature columns.
    """
    df = pd.read_csv(path)

    # Drop unnamed index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

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

    # Ensure each track_id is unique
    metadata = df.drop_duplicates(subset=['track_id']).reset_index(drop=True)
    return metadata.copy()

load_metadata("/Users/chunchiaoyang/Desktop/SoftwareCarpentry/FinalProject/dataset.csv")