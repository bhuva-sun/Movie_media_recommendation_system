"""
Advanced Movie Recommendation System - Training Pipeline
Optimized for TMDB Movies Dataset 2023 (930K+ movies)
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as sk_normalize
from nltk.stem.snowball import SnowballStemmer
import pickle
import json
from pathlib import Path
from ast import literal_eval
import warnings
warnings.filterwarnings('ignore')
from scipy.sparse import issparse


class MovieRecommenderTrainer:
    def __init__(
        self,
        output_dir='./models',
        use_dimensionality_reduction=True,
        n_components=500,
        similarity_top_k=200,
        similarity_min_movies_for_topk=12000,
        similarity_batch_size=256,
    ):
        """
        Initialize the trainer with advanced configurations
        
        Args:
            output_dir: Directory to save trained models
            use_dimensionality_reduction: Use SVD to reduce memory footprint
            n_components: Number of latent features for SVD
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_svd = use_dimensionality_reduction
        self.n_components = n_components
        self.stemmer = SnowballStemmer('english')
        self.similarity_top_k = similarity_top_k
        self.similarity_min_movies_for_topk = similarity_min_movies_for_topk
        self.similarity_batch_size = similarity_batch_size
        
    def load_data(self, data_path):
        """
        Load TMDB dataset from single CSV file
        
        Args:
            data_path: Path to TMDB_movie_dataset_v11.csv
        
        Returns:
            DataFrame with movie data
        """
        print("Loading TMDB dataset...")
        
        # Handle both file path and directory path
        if Path(data_path).is_file():
            df = pd.read_csv(data_path, low_memory=False)
        else:
            # Assume it's a directory
            csv_path = Path(data_path) / 'TMDB_movie_dataset_v11.csv'
            df = pd.read_csv(csv_path, low_memory=False)
        
        print(f"Loaded {len(df)} movies")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def parse_json_column(self, col_data, key='name'):
        """
        Parse JSON-like string columns (genres, keywords, production_companies)
        Handles both string representation and actual lists
        """
        if pd.isna(col_data) or col_data == '' or col_data == '[]':
            return []
        
        try:
            # Try literal_eval first
            parsed = literal_eval(col_data) if isinstance(col_data, str) else col_data
            
            if isinstance(parsed, list):
                # Extract the specified key from each dict
                return [item[key] for item in parsed if isinstance(item, dict) and key in item]
            return []
        except:
            # Fallback: split by comma if it's a simple comma-separated string
            if isinstance(col_data, str):
                return [item.strip() for item in col_data.split(',') if item.strip()]
            return []
    
    def extract_director_from_companies(self, companies_data):
        """
        Extract primary production company as a proxy for director
        (TMDB dataset doesn't have separate crew/director info)
        """
        companies = self.parse_json_column(companies_data)
        return companies[0] if companies else None
    
    def clean_and_engineer_features(self, df, quality_threshold='medium'):
        """
        Advanced feature engineering pipeline for TMDB dataset
        
        Args:
            df: Input DataFrame
            quality_threshold: 'low', 'medium', or 'high' - filters by vote_count
        
        Returns:
            Processed DataFrame
        """
        print("Engineering features...")
        
        # Filter by quality threshold
        thresholds = {
            'low': 5,      # 5+ votes
            'medium': 50,  # 50+ votes (recommended)
            'high': 500    # 500+ votes (high quality only)
        }
        min_votes = thresholds.get(quality_threshold, 50)
        df['is_india'] = df['production_countries'].apply(
            lambda x: any(c.lower() == 'india' for c in self.parse_json_column(x, 'name'))
        )
        df = df[(df['vote_count'] >= min_votes) | df['is_india']].copy()
        print(f"Filtered to {len(df)} movies with {min_votes}+ votes (plus India titles)")
        
        # Filter only released movies
        df = df[df['status'] == 'Released'].copy()
        
        # Parse JSON columns
        print("Parsing genres, keywords, and production companies...")
        df['genres'] = df['genres'].apply(lambda x: self.parse_json_column(x, 'name'))
        df['keywords'] = df['keywords'].apply(lambda x: self.parse_json_column(x, 'name'))
        df['companies'] = df['production_companies'].apply(lambda x: self.parse_json_column(x, 'name'))
        df['countries'] = df['production_countries'].apply(lambda x: self.parse_json_column(x, 'name'))
        
        # Extract primary production company as director proxy
        df['primary_company'] = df['companies'].apply(lambda x: x[0] if x else None)
        
        # Process overview (plot summary)
        df['overview_clean'] = df['overview'].fillna('').astype(str)
        df['overview_words'] = df['overview_clean'].apply(
            lambda x: [word.lower() for word in x.split()[:50]]  # First 50 words
        )
        
        # Process tagline
        df['tagline_clean'] = df['tagline'].fillna('').astype(str)
        df['tagline_words'] = df['tagline_clean'].apply(
            lambda x: [word.lower() for word in x.split()]
        )
        
        # Clean and stem keywords
        df['keywords'] = df['keywords'].apply(
            lambda x: [self.stemmer.stem(kw.lower().replace(" ", "")) for kw in x[:15]]  # Top 15 keywords
        )
        
        # Clean genres
        df['genres'] = df['genres'].apply(
            lambda x: [genre.lower().replace(" ", "") for genre in x]
        )
        
        # Clean companies (top 3, with weight)
        df['companies_weighted'] = df['companies'].apply(
            lambda x: [x[0].lower().replace(" ", "")] * 2 if x and len(x) > 0 else []  # Weight first company
        )
        df['companies_clean'] = df['companies'].apply(
            lambda x: [comp.lower().replace(" ", "") for comp in x[:3]]
        )
        
        # Clean countries
        df['countries_clean'] = df['countries'].apply(
            lambda x: [country.lower().replace(" ", "") for country in x[:2]]
        )
        
        # Create comprehensive soup feature
        df['soup'] = (
            df['keywords'] + 
            df['genres'] * 2 +  # Weight genres more
            df['companies_weighted'] + 
            df['companies_clean'] +
            df['countries_clean'] +
            df['overview_words'] +
            df['tagline_words']
        )
        df['soup'] = df['soup'].apply(lambda x: ' '.join(x) if x else '')
        
        # Filter valid entries
        df = df[df['soup'].str.len() > 20].copy()
        df = df.dropna(subset=['title'])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        # Sort by popularity (combination of vote_average and vote_count)
        df['quality_score'] = df['vote_average'] * np.log1p(df['vote_count'])
        df = df.sort_values('quality_score', ascending=False)
        
        if 'tconst' in df.columns and 'imdb_id' not in df.columns:
          df['imdb_id'] = df['tconst']

        df = df.reset_index(drop=True)
        
        print(f"Processed {len(df)} valid movies")
        return df
    
    def build_tfidf_matrix(self, df):
        """Build TF-IDF matrix with optimized parameters"""
        print("Building TF-IDF matrix...")
        
        # Adjust max_features based on dataset size
        n_movies = len(df)
        if n_movies < 10000:
            max_features = 10000
        elif n_movies < 100000:
            max_features = 15000
        else:
            max_features = 20000
        
        print(f"Using max_features={max_features} for {n_movies} movies")
        
        tfidf = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=3,  # Increased for larger dataset
            max_df=0.7,  # More aggressive filtering
            stop_words='english',
            max_features=max_features,
            sublinear_tf=True  # Use log scaling
        )
        
        tfidf_matrix = tfidf.fit_transform(df['soup'])
        
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        sparsity = (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100
        print(f"Matrix sparsity: {sparsity:.2f}%")
        
        return tfidf_matrix, tfidf
    
    def compute_similarity_matrix(self, tfidf_matrix):
        """Compute similarity with optional dimensionality reduction"""
        if self.use_svd and tfidf_matrix.shape[0] > 1000:
            print(f"Applying SVD dimensionality reduction to {self.n_components} components...")
            
            # Adjust components based on matrix size
            n_components = min(
                self.n_components,
                tfidf_matrix.shape[0] - 1,
                tfidf_matrix.shape[1] - 1
            )
            
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_matrix = svd.fit_transform(tfidf_matrix)
            
            explained_var = svd.explained_variance_ratio_.sum()
            print(f"Explained variance ratio: {explained_var:.3f}")
            print(f"Reduced matrix shape: {reduced_matrix.shape}")
            n_movies = reduced_matrix.shape[0]

            if n_movies >= self.similarity_min_movies_for_topk:
                print(f"Computing top-{self.similarity_top_k} sparse similarities per movie...")
                sparse_sim = self._compute_topk_cosine_similarity_sparse(reduced_matrix, self.similarity_top_k)
                return sparse_sim, svd

            print("Computing cosine similarity (dense)...")
            similarity_matrix = cosine_similarity(reduced_matrix).astype(np.float32)
            return similarity_matrix, svd
        else:
            n_movies = tfidf_matrix.shape[0]
            if n_movies >= self.similarity_min_movies_for_topk:
                print(f"Computing top-{self.similarity_top_k} sparse similarities from TF-IDF...")
                sparse_sim = self._compute_topk_cosine_similarity_sparse_from_tfidf(tfidf_matrix, self.similarity_top_k)
                return sparse_sim, None

            print("Computing cosine similarity (no dimensionality reduction, dense)...")
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float32)
            return similarity_matrix, None

    def _compute_topk_cosine_similarity_sparse_from_tfidf(self, tfidf_matrix, top_k):
        """
        Build a CSR sparse similarity matrix storing only the top-K similarities per row,
        computed directly from (L2-normalized) TF-IDF vectors.
        """
        tfidf_matrix = tfidf_matrix.tocsr().astype(np.float32, copy=False)
        tfidf_norm = sk_normalize(tfidf_matrix, norm='l2', axis=1, copy=False)

        n_movies = tfidf_norm.shape[0]
        k = int(min(top_k + 1, n_movies))  # +1 includes self-similarity

        rows_parts = []
        cols_parts = []
        data_parts = []

        for start in range(0, n_movies, self.similarity_batch_size):
            end = min(start + self.similarity_batch_size, n_movies)
            batch = tfidf_norm[start:end]

            # Sparse dot-product similarity for this batch
            batch_sim = (batch @ tfidf_norm.T).tocsr()

            indptr = batch_sim.indptr
            indices = batch_sim.indices
            data = batch_sim.data

            for local_i in range(0, end - start):
                ptr_start = indptr[local_i]
                ptr_end = indptr[local_i + 1]
                if ptr_start == ptr_end:
                    continue

                row_cols = indices[ptr_start:ptr_end]
                row_data = data[ptr_start:ptr_end]

                if row_data.size > k:
                    kth = row_data.size - k
                    part = np.argpartition(row_data, kth=kth)[-k:]
                    row_cols = row_cols[part]
                    row_data = row_data[part]

                order = np.argsort(row_data)[::-1]
                row_cols = row_cols[order]
                row_data = row_data[order]

                rows_parts.append(np.full(row_cols.shape[0], start + local_i, dtype=np.int32))
                cols_parts.append(row_cols.astype(np.int32, copy=False))
                data_parts.append(row_data.astype(np.float32, copy=False))

            print(f"Processed {end}/{n_movies} rows...")

        rows = np.concatenate(rows_parts) if rows_parts else np.array([], dtype=np.int32)
        cols = np.concatenate(cols_parts) if cols_parts else np.array([], dtype=np.int32)
        data = np.concatenate(data_parts) if data_parts else np.array([], dtype=np.float32)

        sparse_sim = csr_matrix((data, (rows, cols)), shape=(n_movies, n_movies), dtype=np.float32)
        sparse_sim.eliminate_zeros()
        return sparse_sim

    def _compute_topk_cosine_similarity_sparse(self, reduced_matrix, top_k):
        """
        Build a CSR sparse similarity matrix storing only the top-K similarities per row.
        This avoids allocating an (N x N) dense matrix for large N.
        """
        reduced_matrix = reduced_matrix.astype(np.float32, copy=False)
        n_movies = reduced_matrix.shape[0]
        k = int(min(top_k + 1, n_movies))  # +1 includes self-similarity

        norms = np.linalg.norm(reduced_matrix, axis=1, keepdims=True).astype(np.float32)
        norms[norms == 0] = 1.0
        reduced_norm = reduced_matrix / norms

        rows_parts = []
        cols_parts = []
        data_parts = []

        print(f"Top-K sparse mode: N={n_movies:,}, K={k:,}, batch={self.similarity_batch_size:,}")

        xt = reduced_norm.T
        for start in range(0, n_movies, self.similarity_batch_size):
            end = min(start + self.similarity_batch_size, n_movies)
            batch = reduced_norm[start:end]

            sims = batch @ xt  # (batch_size, n_movies)

            # Grab top-k indices efficiently
            kth = n_movies - k
            idx_part = np.argpartition(sims, kth=kth, axis=1)[:, -k:]
            scores_part = np.take_along_axis(sims, idx_part, axis=1)

            order = np.argsort(scores_part, axis=1)[:, ::-1]
            top_idx = np.take_along_axis(idx_part, order, axis=1)
            top_scores = np.take_along_axis(scores_part, order, axis=1)

            row_idx = np.repeat(np.arange(start, end, dtype=np.int32), k)

            rows_parts.append(row_idx)
            cols_parts.append(top_idx.astype(np.int32, copy=False).ravel())
            data_parts.append(top_scores.astype(np.float32, copy=False).ravel())

            if end % (self.similarity_batch_size * 10) == 0:
                print(f"Processed {end}/{n_movies} rows...")

        rows = np.concatenate(rows_parts)
        cols = np.concatenate(cols_parts)
        data = np.concatenate(data_parts)

        sparse_sim = csr_matrix((data, (rows, cols)), shape=(n_movies, n_movies), dtype=np.float32)
        sparse_sim.eliminate_zeros()
        return sparse_sim
    
    def save_model(self, df, similarity_matrix, tfidf_vectorizer, svd_model=None):
        """Save all model artifacts efficiently"""
        print("Saving model artifacts...")
        
        # Save metadata DataFrame (essential columns only)
        metadata_df = df[[
            'id', 'title', 'release_date', 'primary_company', 
            'genres', 'vote_average', 'vote_count', 'popularity',
            'overview', 'imdb_id', 'poster_path'
        ]].copy()
        
        metadata_df.to_parquet(
            self.output_dir / 'movie_metadata.parquet',
            compression='gzip',
            index=True
        )
        
        # Save similarity matrix
        print("Saving similarity matrix...")
        if issparse(similarity_matrix):
            sparse_sim = similarity_matrix.tocsr().astype(np.float32)
            save_npz(self.output_dir / 'similarity_matrix.npz', sparse_sim)
            print(f"Saved as sparse top-K matrix (data: {sparse_sim.data.nbytes / 1024**2:.1f} MB)")
            # Remove any stale dense artifact
            dense_path = self.output_dir / "similarity_matrix.npy"
            if dense_path.exists():
                try:
                    dense_path.unlink()
                except OSError:
                    pass
        elif similarity_matrix.size > 10000000:  # > 10M elements
            # Save as sparse for very large matrices (may still be large in practice)
            sparse_sim = csr_matrix(similarity_matrix)
            save_npz(self.output_dir / 'similarity_matrix.npz', sparse_sim)
            print(f"Saved as sparse matrix (size: {sparse_sim.data.nbytes / 1024**2:.1f} MB)")
        else:
            np.save(self.output_dir / 'similarity_matrix.npy', similarity_matrix)
            print(f"Saved as dense matrix (size: {similarity_matrix.nbytes / 1024**2:.1f} MB)")
        
        # Save title to index mapping
        title_to_idx = pd.Series(df.index, index=df['title']).to_dict()
        with open(self.output_dir / 'title_to_idx.json', 'w') as f:
            json.dump(title_to_idx, f)
        
        # Save TF-IDF vectorizer
        with open(self.output_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        
        # Save SVD model if used
        if svd_model:
            with open(self.output_dir / 'svd_model.pkl', 'wb') as f:
                pickle.dump(svd_model, f)
        
        # Save configuration
        config = {
            'n_movies': len(df),
            'use_svd': self.use_svd,
            'n_components': self.n_components if svd_model else None,
            'matrix_shape': similarity_matrix.shape,
            'dataset': 'TMDB 2023 (930K movies)'
        }
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Model saved to {self.output_dir}")
        
        # Print summary
        total_size = sum(
            f.stat().st_size 
            for f in self.output_dir.iterdir() 
            if f.is_file()
        ) / 1024**2
        print(f"Total model size: {total_size:.1f} MB")
    
    def train(self, data_path, quality_threshold='medium', max_movies=None):
        """
        Complete training pipeline
        
        Args:
            data_path: Path to CSV file or directory containing it
            quality_threshold: 'low', 'medium', or 'high'
            max_movies: Limit number of movies (None = all)
        """
        print("="*80)
        print("🎬 TMDB Movie Recommendation System Training")
        print("="*80)
        
        # Load data
        df = self.load_data(data_path)
        
        # Feature engineering
        df = self.clean_and_engineer_features(df, quality_threshold)
        
        # Limit dataset if specified
        if max_movies and len(df) > max_movies:
            if 'is_india' in df.columns:
                india_ratio = 0.3
                india_quota = int(max_movies * india_ratio)
                india_df = df[df['is_india']].head(india_quota)
                non_india_df = df[~df['is_india']].head(max_movies - len(india_df))
                df = pd.concat([india_df, non_india_df], ignore_index=True)
                print(
                    f"Limited to top {len(df)} movies with Bollywood/India boost: "
                    f"{len(india_df)} India titles + {len(non_india_df)} others"
                )
            else:
                df = df.head(max_movies)
                print(f"Limited to top {max_movies} movies by quality score")
        
        # Build TF-IDF matrix
        tfidf_matrix, tfidf_vectorizer = self.build_tfidf_matrix(df)
        
        # Compute similarity
        similarity_matrix, svd_model = self.compute_similarity_matrix(tfidf_matrix)
        
        # Save everything
        self.save_model(df, similarity_matrix, tfidf_vectorizer, svd_model)
        
        print("="*80)
        print("✅ Training completed successfully!")
        print("="*80)
        
        return df, similarity_matrix


# Example usage
if __name__ == "__main__":
    
    # Downloaded dataset
    path = "./TMDB  IMDB Movies Dataset.csv"
    
    # Configuration based on your needs:
    
    # For FULL dataset (930K+ movies) - Requires ~16GB RAM
    # trainer = MovieRecommenderTrainer(
    #     output_dir='./models_full',
    #     use_dimensionality_reduction=True,
    #     n_components=400
    # )
    # df, sim_matrix = trainer.train(path, quality_threshold='low')
    
    # For HIGH QUALITY dataset (~100K movies) - Recommended
    trainer = MovieRecommenderTrainer(
        output_dir='./models',
        use_dimensionality_reduction=True,
        n_components=500
    )
    df, sim_matrix = trainer.train(
        path, 
        quality_threshold='medium',  # 50+ votes
        max_movies=50000  # Top 100K by quality
    )
    
    # For MEDIUM dataset (~10K movies) - Fast training
    # trainer = MovieRecommenderTrainer(
    #     output_dir='./models_medium',
    #     use_dimensionality_reduction=False
    # )
    # df, sim_matrix = trainer.train(path, quality_threshold='high', max_movies=10000)
    
    print(f"\n📊 Final Statistics:")
    print(f"   Movies in model: {len(df):,}")
    print(f"   Similarity matrix: {sim_matrix.shape}")
    print(f"   Memory usage: {sim_matrix.nbytes / 1024**2:.1f} MB")

