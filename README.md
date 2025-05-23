# Music Recommendation Engine

A content-based and hybrid music recommender that uses track metadata (genre + audio features) and popularity as a proxy for user interactions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites & Installation](#prerequisites--installation)
4. [Usage](#usage)
5. [Running Unit Tests](#running-unit-tests)
6. [Module Descriptions](#module-descriptions)
7. [Configuration & Parameters](#configuration--parameters)
8. [Example](#example)
9. [Requirements](#requirements)

---

## Project Overview

Millions of tracks on streaming platforms make discovery challenging.
This project builds a lightweight recommendation engine that:

* Loads and cleans track metadata
* Encodes genre as one-hot and scales audio features
* Applies optional feature weighting and PCA
* Computes content similarity and blends with track popularity (hybrid)
* Outputs top-N recommended tracks for a given seed track

---

## Repository Structure

```text
├── dataset.csv                # Metadata CSV (e.g., dataset_clean.csv)
├── music_recommender/         # Core modules
│   ├── data_loader.py         # Load & dedupe metadata
│   └── feature_engineering.py # Build feature matrix & hybrid scoring
├── recommender.py             # Runs the hybrid recommender
├── tests/                     # pytest unit tests
│   ├── test_data_loader.py
│   └── test_feature_engineering.py
├── .gitignore
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Prerequisites & Installation

1. **Clone the repository**

   ```bash
   git clone git@github.com:YOUR_USERNAME/music_recommender.git
   cd music_recommender
   ```
2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .\venv\Scripts\activate
   ```
3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Install the project package

To ensure Python can find and import the `music_recommender` modules, install your project into the environment:

1. **Install in editable mode** (recommended):

   ```bash
   pip install -e .
   ```

   This links the local source into your environment so that `import music_recommender` works automatically.

2. **Or use PYTHONPATH** (alternative):

   ```bash
   export PYTHONPATH=\"$PWD\"
   ```

   This tells Python to treat your project root as an importable package location.

After one of these steps, you can run scripts and tests without import errors.

---

## Usage

Run the hybrid recommender via the CLI:

```bash
python scripts/recommender.py \
  --metadata    dataset.csv \
  --seed        <TRACK_ID> \
  --top-n       10 \
  --genre-weight 1.0 \
  --audio-weight 1.0 \
  --pca-components None \
  --alpha       0.5 \
  --output      recommendations.csv \
  --log         logs/recommender.log
```

Results will be printed to stdout and saved if `--output` is provided.

---

## Running Unit Tests

After installation, run the full test suite with pytest:

```bash
pytest -q
```

Make sure you have `pytest` installed (it’s included in `requirements.txt`).

---

## Module Descriptions

### `music_recommender/data_loader.py`

* **`load_metadata(path)`**: Reads a metadata CSV, drops `Unnamed: 0`, removes duplicates (same artists+album+track), verifies required columns, ensures unique `track_id`.

### `music_recommender/feature_engineering.py`

* **`build_content_matrix(metadata, genre_weight, audio_weight, pca_components)`**:

  * One-hot encodes `track_genre`, scales numeric features, applies weights, optionally reduces via PCA.
  * Returns a sparse feature matrix, ordered `track_ids`, and `feature_names`.
* **`compute_content_similarity(seed_id, content_mat, track_ids)`**: Cosine similarity between a seed track and all others (seed excluded).
* **`hybrid_scores(content_sims, popularity, alpha)`**: Normalizes popularity and linearly blends with content similarity.
* **`recommend_hybrid(...)`**: End-to-end hybrid recommendation, returning top-N `track_id`s (excluding seed).

### `scripts/recommender.py`

* Parses CLI args for metadata path, seed, weights, PCA, alpha, top-N, log, and output.
* Loads metadata, calls `recommend_hybrid`, prints a report (seed info + recommendation table).

---

## Configuration & Parameters

| Flag               | Description                                                                             | Default                |
| ------------------ | --------------------------------------------------------------------------------------- | ---------------------- |
| `--metadata`, `-m` | Path to the deduped metadata CSV                                                        | **(required)**         |
| `--seed`, `-s`     | Seed `track_id` to base recommendations on                                              | **(required)**         |
| `--top-n`, `-n`    | Number of recommendations to generate                                                   | `10`                   |
| `--genre-weight`   | Multiplier for genre one-hot features                                                   | `1.0`                  |
| `--audio-weight`   | Multiplier for numeric/audio features                                                   | `1.0`                  |
| `--pca-components` | Reduce features to this many PCA components (int) or `None` for no PCA                  | `None`                 |
| `--alpha`          | Hybrid blend: `alpha` · content\_similarity + (1-`alpha`) · popularity                  | `0.5`                  |
| `--output`, `-o`   | Path to save recommendations CSV (columns: `track_id, artists, album_name, track_name`) | *(prints to stdout)*   |
| `--log`, `-l`      | Path for log file (INFO & errors)                                                       | `logs/recommender.log` |

---

## Example

```bash
python scripts/recommender.py \
  -m dataset.csv \
  -s 3n3Ppam7vgaVa1iaRUc9Lp \
  -n 5 \
  --genre-weight 2.0 \
  --audio-weight 1.5 \
  --pca-components 8 \
  --alpha 0.7 \
  -o results/top5.csv \
  -l logs/run1.log
```

---

## Requirements

* Python 3.8+
* pandas
* numpy
* scipy
* scikit-learn
* pytest

Install via:

```bash
pip install -r requirements.txt
```

---
*spotify dataset source: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset*

*Happy recommending!*
