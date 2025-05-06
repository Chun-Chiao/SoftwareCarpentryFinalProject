import pytest
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from music_recommender.feature_engineering import build_content_matrix, recommend_similar_tracks


def make_metadata_df():
    # Two example tracks with distinct genres and feature values
    return pd.DataFrame([
        {
            'track_id': 't1', 'track_genre': 'rock',
            'danceability': 0.5, 'energy': 0.6, 'key': 1, 'loudness': -5,
            'mode': 1, 'speechiness': 0.1, 'acousticness': 0.2,
            'instrumentalness': 0.0, 'liveness': 0.05, 'valence': 0.3,
            'tempo': 120.0, 'duration_ms': 180000, 'popularity': 50,
            'artists': 'Artist1', 'album_name': 'Album1', 'track_name': 'Track1'
        },
        {
            'track_id': 't2', 'track_genre': 'pop',
            'danceability': 0.8, 'energy': 0.7, 'key': 5, 'loudness': -4,
            'mode': 1, 'speechiness': 0.05, 'acousticness': 0.1,
            'instrumentalness': 0.01, 'liveness': 0.02, 'valence': 0.7,
            'tempo': 130.0, 'duration_ms': 200000, 'popularity': 60,
            'artists': 'Artist2', 'album_name': 'Album2', 'track_name': 'Track2'
        }
    ])


def test_build_content_matrix_outputs():
    df = make_metadata_df()
    mat, track_ids, feature_names = build_content_matrix(df)
    # Expect two rows (tracks)
    assert mat.shape[0] == 2
    # Expect number of features = number of unique genres + numeric features
    # Two genres -> 2; numeric_cols defined in module = 13
    assert mat.shape[1] == 2 + 13
    # Track IDs order preserved
    assert track_ids == ['t1', 't2']
    # Feature names include genre prefixes and numeric cols
    assert all(name.startswith('genre_') for name in feature_names[:2])
    for col in ['danceability','energy','tempo','popularity']:
        assert col in feature_names
    # Matrix entries should be finite numbers
    arr = mat.toarray()
    assert np.isfinite(arr).all()


def test_missing_numeric_columns_raises():
    df = make_metadata_df().drop(columns=['tempo', 'popularity'])
    with pytest.raises(ValueError) as exc:
        build_content_matrix(df)
    msg = str(exc.value)
    assert 'Missing numeric metadata columns' in msg
    assert 'tempo' in msg and 'popularity' in msg


def test_recommend_similar_tracks_topology():
    df = make_metadata_df()
    mat, track_ids, _ = build_content_matrix(df)
    # t1 should get t2 as most similar, and vice versa
    recs_for_t1 = recommend_similar_tracks('t1', mat, track_ids, [], top_n=1)
    recs_for_t2 = recommend_similar_tracks('t2', mat, track_ids, [], top_n=1)
    assert recs_for_t1 == ['t2']
    assert recs_for_t2 == ['t1']


def test_recommend_invalid_track_raises():
    df = make_metadata_df()
    mat, track_ids, _ = build_content_matrix(df)
    with pytest.raises(ValueError):
        recommend_similar_tracks('invalid_id', mat, track_ids, [], top_n=3)
