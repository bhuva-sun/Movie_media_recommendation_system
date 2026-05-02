#!/usr/bin/env python3
"""Training script to (re)build the recommendation model used by the web app.

This script:
- downloads the TMDB dataset via Kaggle
- trains a model with ~20K high‑quality movies (including Bollywood and other global cinema)
- writes artifacts into `training/models`, which is what the Django app loads by default.
"""

from pathlib import Path
from typing import Optional

import nltk

from training.train import MovieRecommenderTrainer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "training" / "models"
DATASET_HANDLE = "asaniczka/tmdb-movies-dataset-2023-930k-movies"
DATASET_CSV_NAME = "TMDB_movie_dataset_v11.csv"


def main() -> None:
    # Download NLTK data
    print("Downloading NLTK data...")
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

    # Download dataset (TMDB, includes Hollywood, Bollywood, and other industries)
    dataset_csv = _get_cached_dataset_csv()
    if dataset_csv:
        print(f"Using cached TMDB dataset CSV: {dataset_csv}")
        data_path: str = str(dataset_csv)
    else:
        # Fall back to downloading from Kaggle (may require network access).
        import kagglehub

        print("Downloading TMDB dataset (this may take a moment)...")
        data_path = kagglehub.dataset_download(DATASET_HANDLE)
        print(f"Dataset downloaded to: {data_path}")

    # Configure trainer:
    # - output_dir: training/models so the Django app picks it up
    # - use_dimensionality_reduction=True to keep memory manageable for 20K movies
    print("\nTraining model (20K movies, global catalog)...")
    trainer = MovieRecommenderTrainer(
        output_dir=str(MODEL_DIR),
        use_dimensionality_reduction=False,
        similarity_top_k=120,
        similarity_batch_size=128,
    )

    # quality_threshold='medium' (50+ votes) keeps popular titles while
    # still covering a broad range of international / Bollywood movies.
    df, sim_matrix = trainer.train(
        data_path,
        quality_threshold="medium",
        max_movies=20000,
    )

    print(
        f"\nTraining complete! Model saved to {MODEL_DIR} "
        f"with {len(df):,} movies and matrix shape {sim_matrix.shape}."
    )


def _get_cached_dataset_csv() -> Optional[Path]:
    versions_dir = (
        Path.home()
        / ".cache"
        / "kagglehub"
        / "datasets"
        / "asaniczka"
        / "tmdb-movies-dataset-2023-930k-movies"
        / "versions"
    )
    if not versions_dir.exists():
        return None

    candidates = list(versions_dir.glob(f"*/{DATASET_CSV_NAME}"))
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


if __name__ == "__main__":
    main()
