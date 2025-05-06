"""
scripts/recommender.py

Command-line interface for the content-based recommendation engine.
"""
import argparse
import logging
import pandas as pd
from music_recommender.data_loader import load_metadata
from music_recommender.feature_engineering import build_content_matrix, recommend_similar_tracks


def main():
    parser = argparse.ArgumentParser(
        description="Generate content-based music recommendations from track metadata."
    )
    parser.add_argument(
        '--metadata', '-m',
        required=True,
        help='Path to the track metadata CSV file'
    )
    parser.add_argument(
        '--seed', '-s',
        required=True,
        help='Seed track ID for which to find similar tracks'
    )
    parser.add_argument(
        '--top-n', '-n',
        type=int,
        default=10,
        help='Number of recommendations to generate (default: 10)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Optional path to save recommendations as CSV'
    )
    parser.add_argument(
        '--log', '-l',
        default='logs/recommender.log',
        help='Path to log file (default: logs/recommender.log)'
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    logging.info(f"Starting recommendations for seed track: {args.seed}")

    # Load metadata and build content features
    metadata = load_metadata(args.metadata)
    content_mat, track_ids, feature_names = build_content_matrix(metadata)
    logging.info(f"Built content matrix with {content_mat.shape[0]} tracks and {content_mat.shape[1]} features")

    # Generate recommendations
    try:
        recs = recommend_similar_tracks(
            track_id=args.seed,
            content_mat=content_mat,
            track_ids=track_ids,
            feature_names=feature_names,
            top_n=args.top_n
        )
    except ValueError as e:
        logging.error(str(e))
        print(f"Error: {e}")
        return

    logging.info(f"Recommendations: {recs}")

    # Prepare output DataFrame
    df_out = pd.DataFrame({
        'seed_track_id': [args.seed] * len(recs),
        'recommended_track_id': recs
    })

    if args.output:
        df_out.to_csv(args.output, index=False)
        print(f"Saved recommendations to {args.output}")
    else:
        print("Recommendations:")
        print(df_out.to_string(index=False))


if __name__ == '__main__':
    main()
