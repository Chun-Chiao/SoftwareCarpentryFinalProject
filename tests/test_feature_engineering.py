import pytest
import pandas as pd
import numpy as np

from music_recommender.feature_engineering import (
    build_content_matrix,
    compute_content_similarity,
    hybrid_scores,
    recommend_hybrid
)


def make_metadata_df():
    # Create a simple metadata DataFrame with two distinct genres and popularity
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
            'tempo': 130.0, 'duration_ms': 200000, 'popularity': 100,
            'artists': 'Artist2', 'album_name': 'Album2', 'track_name': 'Track2'
        }
    ])


def test_build_content_matrix_defaults():
    df = make_metadata_df()
    mat, track_ids, feature_names = build_content_matrix(df)
    # Expect 2 tracks and 2 genres + 13 numeric = 15 features
    assert mat.shape == (2, 2 + 13)
    assert track_ids == ['t1', 't2']
    # Feature names: first two start with 'genre_' then numeric
    assert all(name.startswith('genre_') for name in feature_names[:2])
    for col in ['danceability', 'energy', 'tempo', 'popularity']:
        assert col in feature_names


def test_build_content_matrix_weights_and_pca():
    df = make_metadata_df()
    # Use weights and reduce to 2 components
    mat, track_ids, feature_names = build_content_matrix(
        df, genre_weight=2.0, audio_weight=0.5, pca_components=2
    )
    # After PCA, shape should be (2,2)
    assert mat.shape == (2, 2)
    assert feature_names == ['pc_1', 'pc_2']
    # Values should be finite
    arr = mat.toarray()
    assert np.all(np.isfinite(arr))


def test_compute_content_similarity_and_error():
    df = make_metadata_df()
    mat, track_ids, _ = build_content_matrix(df)
    sims = compute_content_similarity('t1', mat, track_ids)
    # Self similarity excluded
    assert sims[track_ids.index('t1')] == -np.inf
    # Similarity vector length correct
    assert sims.shape == (2,)
    # Invalid seed raises
    with pytest.raises(ValueError):
        compute_content_similarity('invalid', mat, track_ids)


def test_hybrid_scores():
    # simulate content sims and popularity
    content_sims = np.array([0.5, 0.2])
    popularity = np.array([50.0, 100.0])
    hybrid = hybrid_scores(content_sims, popularity, alpha=0.5)
    # popularity normalized: [0,1]
    # hybrid = 0.5*[0.5,0.2] + 0.5*[0,1] = [0.25, 0.6]
    expected = np.array([0.25, 0.6])
    assert pytest.approx(hybrid, rel=1e-6) == expected


def test_recommend_hybrid_basic():
    df = make_metadata_df()
    # recommend_hybrid uses default weights and no PCA
    recs = recommend_hybrid(
        seed_id='t1',
        content_mat=None,
        track_ids=None,
        metadata=df,
        genre_weight=1.0,
        audio_weight=1.0,
        pca_components=None,
        alpha=0.5,
        top_n=1
    )
    # Only one other track, so recommendation should be ['t2']
    assert recs == ['t2']

    recs_all = recommend_hybrid(
        seed_id='t1', metadata=df, top_n=2
    )
    # top_n > available yields both but seed excluded then the seed itself not returned
    assert set(recs_all) == {'t2'}


def test_recommend_hybrid_missing_seed():
    df = make_metadata_df()
    with pytest.raises(ValueError):
        recommend_hybrid(
            seed_id='invalid', metadata=df, top_n=1
        )
