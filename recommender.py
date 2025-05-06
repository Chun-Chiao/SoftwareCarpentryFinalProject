"""
scripts/recommender.py

CLI for hybrid content+popularity recommendation with feature weighting and PCA.
"""
import argparse
import logging
import pandas as pd
from music_recommender.data_loader import load_metadata
from music_recommender.feature_engineering import recommend_hybrid


def main():
    parser = argparse.ArgumentParser(
        description="Generate content-based hybrid music recommendations with weighting and PCA.")
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
        '--genre-weight',
        type=float,
        default=1.0,
        help='Weight for genre features (default: 1.0)'
    )
    parser.add_argument(
        '--audio-weight',
        type=float,
        default=1.0,
        help='Weight for numeric audio features (default: 1.0)'
    )
    parser.add_argument(
        '--pca-components',
        type=int,
        default=None,
        help='Number of PCA components to reduce (default: None)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Hybrid weight: content vs. popularity (0..1, default:0.5)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Optional path to save recommendations as CSV'
    )
    parser.add_argument(
        '--log', '-l',
        default='logs/recommender.log',
        help='Path to log file'
    )
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    logging.info(
        f"Starting hybrid recommendations: seed={args.seed}, alpha={args.alpha}")

    # Load metadata
    metadata = load_metadata(args.metadata)
    logging.info(f"Loaded {len(metadata)} tracks from metadata")

    # Generate hybrid recommendations
    recs = recommend_hybrid(
        seed_id=args.seed,
        content_mat=None,
        track_ids=None,
        metadata=metadata,
        genre_weight=args.genre_weight,
        audio_weight=args.audio_weight,
        pca_components=args.pca_components,
        alpha=args.alpha,
        top_n=args.top_n
    )
    logging.info(f"Generated {len(recs)} recommendations")

    # Print seed track info
    seed_row = metadata[metadata['track_id'] == args.seed].iloc[0]
    print("Seed Track:")
    print(f"  ID: {args.seed}")
    print(f"  Artist: {seed_row['artists']}")
    print(f"  Album: {seed_row['album_name']}")
    print(f"  Title: {seed_row['track_name']}\n")

    # Build and print recommendations table
    rec_rows = []
    for rid in recs:
        row = metadata[metadata['track_id'] == rid].iloc[0]
        rec_rows.append({
            'track_id': rid,
            'artists': row['artists'],
            'album_name': row['album_name'],
            'track_name': row['track_name']
        })
    df_recs = pd.DataFrame(
        rec_rows,
        columns=[
            'track_id',
            'artists',
            'album_name',
            'track_name'])
    print("Recommended Tracks:")
    print(df_recs.to_string(index=False))

    # Optionally save
    if args.output:
        df_recs.to_csv(args.output, index=False)
        print(f"Saved recommendations to {args.output}")


if __name__ == '__main__':
    main()
