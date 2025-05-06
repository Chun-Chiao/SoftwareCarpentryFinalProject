import pytest
import pandas as pd
from pathlib import Path

from music_recommender.data_loader import load_interactions, load_metadata


def test_load_interactions_success(tmp_path):
    # Create a temp CSV with expected columns and extra column
    csv_content = """
user_id,track_id,rating,extra
u1,t1,5,foo
u2,t2,3,bar
"""
    file = tmp_path / "interactions.csv"
    file.write_text(csv_content.strip())

    df = load_interactions(str(file))
    # Should only have user_id, track_id, rating columns
    assert list(df.columns) == ['user_id', 'track_id', 'rating']
    assert df.shape == (2, 3)
    # Values preserved correctly
    assert df['user_id'].tolist() == ['u1', 'u2']
    assert df['track_id'].tolist() == ['t1', 't2']
    assert df['rating'].tolist() == [5, 3]


def test_load_interactions_missing_column(tmp_path):
    # Missing 'rating' column
    csv_content = """
user_id,track_id,extra
u1,t1,foo
"""
    file = tmp_path / "bad_interactions.csv"
    file.write_text(csv_content.strip())

    with pytest.raises(ValueError) as exc:
        load_interactions(str(file))
    assert "Missing columns in interactions file" in str(exc.value)


def test_load_metadata_success(tmp_path):
    # Build a DataFrame with two rows, one duplicate on artists+album+track_name
    data = [
        {
            'track_id': 't1', 'artists': 'A1', 'album_name': 'Album', 'track_name': 'Song',
            'popularity': 50, 'duration_ms': 200000, 'explicit': False,
            'danceability': 0.5, 'energy': 0.6, 'key': 1, 'loudness': -5,
            'mode': 1, 'speechiness': 0.1, 'acousticness': 0.2,
            'instrumentalness': 0.0, 'liveness': 0.05, 'valence': 0.3,
            'tempo': 120, 'time_signature': 4, 'track_genre': 'rock'
        },
        {
            'track_id': 't2', 'artists': 'A1', 'album_name': 'Album', 'track_name': 'Song',
            'popularity': 60, 'duration_ms': 210000, 'explicit': True,
            'danceability': 0.7, 'energy': 0.8, 'key': 2, 'loudness': -4,
            'mode': 1, 'speechiness': 0.05, 'acousticness': 0.1,
            'instrumentalness': 0.01, 'liveness': 0.02, 'valence': 0.7,
            'tempo': 130, 'time_signature': 4, 'track_genre': 'rock'
        }
    ]
    df_orig = pd.DataFrame(data)
    file = tmp_path / "metadata.csv"
    df_orig.to_csv(file, index=False)

    df = load_metadata(str(file))
    # Duplicate on artists+album+track_name should remove one row
    assert df.shape[0] == 1
    # Columns present and in correct order
    expected_cols = [
        'track_id', 'artists', 'album_name', 'track_name',
        'popularity', 'duration_ms', 'explicit', 'danceability',
        'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence',
        'tempo', 'time_signature', 'track_genre'
    ]
    assert list(df.columns) == expected_cols
    # The remaining track_id should be the first occurrence 't1'
    assert df['track_id'].iloc[0] == 't1'


def test_load_metadata_missing_columns(tmp_path):
    # Create metadata missing 'track_genre'
    df_bad = pd.DataFrame([
        {'track_id': 't1', 'artists': 'A1', 'album_name': 'Album', 'track_name': 'Song'}
    ])
    file = tmp_path / "bad_meta.csv"
    df_bad.to_csv(file, index=False)

    with pytest.raises(ValueError) as exc:
        load_metadata(str(file))
    assert 'Missing metadata columns' in str(exc.value)
