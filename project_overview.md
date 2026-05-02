# Movie Recommendation System Project Overview

This document provides a comprehensive overview of the Movie Recommendation System, detailing its architecture, components, models, and deployment configuration.

## 1. Project Architecture

The project is built as a web application utilizing the **Django** framework, integrated with a Machine Learning pipeline for providing movie recommendations. It operates through precomputed similarity matrices for fast, real-time response.

- **Backend Framework:** Django (Python)
- **Machine Learning Libraries:** Pandas, NumPy, SciPy (for sparse matrices), NLTK (for NLP tasks)
- **Database:** SQLite3 (local development context, managed by Django, though main recommendations act on serialized ML artifacts)
- **Deployment:** Pre-configured for Render (`render.yaml`) and Heroku-like Platform-as-a-Service environments (`Procfile` using Gunicorn)

## 2. Directory Structure & Key Files

### `/recommender/`
This is the core Django app for this project.
- **`views.py`**: The main logic controller. Contains the `MovieRecommender` class. It manages in-memory data loading logic (lazy loads ML models in a background thread to bypass slow boot times), handles fuzzy matching of user queries via `difflib`, calculates similarity scores, and serves the recommendation views.
- **`urls.py`**: Defines app routes including serving the HTML pages, API endpoints for searching movies (`/api/search/`), checking model load status (`/api/model-status/`), and health checks (`/api/health/`).
- **Templates**: Likely contains the frontend HTML views for rendering the search bar and recommendation outcomes (`index.html`, `result.html`).

### `/movie_recommendation/`
This is the generic Django project configuration folder containing standard generated files like `settings.py`, `urls.py`, and configurations for running via WSGI/ASGI.

### `/models/`
Stores the artifacts generated after training the model. The Django app directly loads these into memory.
- `movie_metadata.parquet`: Compressed metadata representation of the trained movie dataset.
- `similarity_matrix.npz` / `similarity_matrix.npy`: A pre-computed dense or sparse matrix mapping distance/similarities between different movies.
- `tfidf_vectorizer.pkl`: Pickled Scikit-Learn TF-IDF representation used during feature extraction of movie descriptions/genres.
- `title_to_idx.json`: Maps string titles to indices of the similarity matrix and dataset to speed up queries.
- `config.json`: Saves dataset configuration specifics (e.g. number of movies loaded).

### `/training/`
Contains the source code used to generate the models.
- **`run_training.py`**: A quick-start wrapper script. It downloads necessary NLTK corpora (punkt, stopwords) and fetches the **TMDB movie dataset** from Kaggle (`asaniczka/tmdb-movies-dataset-2023-930k-movies`) via `kagglehub`. It configures a fast training loop limiting to a certain subset (e.g., top 10K movies with 500+ votes for fast development).
- **`train.py`**: Encapsulates the `MovieRecommenderTrainer` class that processes the raw TMDB Dataframe, calculates feature scores using TF-IDF, prepares the similarity matrix, and exports the artifacts into `/models/`.
- **`infer.py`**: A testing/inference script to validate recommendations natively from CLI before deploying them to the web server.

### Configuration Files
- **`Procfile`**: Setup for deployment environments that read Procfiles, instructing them to run migrations and start the server using Gunicorn (`gunicorn movie_recommendation.wsgi:application --log-file - --log-level info`).
- **`render.yaml`**: Configuration for automatic deployment to **Render**. Specifies Python 3.10.13, builds using `pip install` and `collectstatic`, provisions environmental variables, handles migrations, and defines the health-check path `/api/health/`.

## 3. How the Recommendation Engine Works

1. **Background Loading**: When the web application starts, it creates a background daemon thread that eagerly loads the heavy `.parquet` files and similarity matrices into memory. This ensures the web application starts quickly without timing out.
2. **Querying**: 
   - A user types a query on the frontend. The system provides autocompletion using the `/api/search/` endpoint.
   - Upon submission, if the exact title is not found, `difflib` is used to execute a fuzzy match searching for the nearest valid movie.
3. **Similarity Calculation**:
   - The matched movie's internal ID is cross-referenced with the pre-calculated `similarity_matrix`.
   - The system retrieves the arrays of scores representing how similar this movie is to all others.
   - It filters the result based on specified criteria (like `min_rating`), removes the source movie itself, and limits the output to top `n` recommendations (default `15`).
4. **Rich Metadata View**: The view formats information seamlessly, exposing fields like genre, vote scores, TMDB poster, and generating automated links to Google and IMDB.

## 4. Getting Started Locally

1. Setup a standard Python Virtual Environment and install dependencies (usually via `requirements.txt`).
2. If models are missing in `/models/`, execute `python run_training.py` to trigger the Kaggle dataset download and initiate model computation locally.
3. Apply Django database migrations: `python manage.py migrate`.
4. Run the local development server: `python manage.py runserver`.

## 5. Auth and Watchlist Configuration

### Firebase (Client Auth)
Set these environment variables if you want to override defaults:

- `FIREBASE_API_KEY`
- `FIREBASE_AUTH_DOMAIN`
- `FIREBASE_PROJECT_ID`
- `FIREBASE_STORAGE_BUCKET`
- `FIREBASE_MESSAGING_SENDER_ID`
- `FIREBASE_APP_ID`
- `FIREBASE_MEASUREMENT_ID`

### Supabase (Watchlist Storage)
Set:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

Create table in Supabase SQL editor:

```sql
create extension if not exists pgcrypto;

create table if not exists public.watchlist_items (
    id uuid primary key default gen_random_uuid(),
    firebase_uid text not null,
    title text not null,
    poster_url text,
    release_date text,
    rating text,
    created_at timestamptz not null default now(),
    unique (firebase_uid, title)
);

create index if not exists watchlist_items_firebase_uid_idx
    on public.watchlist_items (firebase_uid);
```
