"""
Movie Recommendation System Views
Integrates with advanced TMDB model training system
"""
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional
from difflib import get_close_matches
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import json
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .supabase_client import add_watchlist_item, list_watchlist, remove_watchlist_item

logger = logging.getLogger(__name__)

# Global cache for recommender system
_RECOMMENDER = None
_MODEL_LOADING = False
_MODEL_LOAD_PROGRESS = 0
_LOADING_THREAD = None
_LOAD_ERROR = None
_TRENDING_CACHE = {"timestamp": 0.0, "movies": []}
_TRENDING_CACHE_TTL_SECONDS = 300


def _build_frontend_settings() -> Dict[str, Dict[str, str]]:
    """Build Firebase config for frontend templates."""
    firebase_config = dict(getattr(settings, "FIREBASE_CONFIG", {}))
    return {"firebase_config": firebase_config}


def _json_payload(request) -> Dict:
    """Safely parse JSON request body."""
    if not request.body:
        return {}
    try:
        return json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _fetch_trending_movies(limit: int = 10) -> List[Dict[str, str]]:
    """Fetch live trending movies from TMDB and cache briefly."""
    now = time.time()
    cached_movies = _TRENDING_CACHE.get("movies", [])
    if cached_movies and now - _TRENDING_CACHE.get("timestamp", 0.0) < _TRENDING_CACHE_TTL_SECONDS:
        return cached_movies[:limit]

    tmdb_api_key = os.environ.get("TMDB_API_KEY", "").strip()
    tmdb_read_token = os.environ.get("TMDB_READ_ACCESS_TOKEN", "").strip()
    if not tmdb_api_key and not tmdb_read_token:
        return cached_movies[:limit]

    endpoint = "https://api.themoviedb.org/3/trending/movie/day"
    if tmdb_api_key and not tmdb_read_token:
        endpoint = endpoint + "?" + urlparse.urlencode({"api_key": tmdb_api_key})

    request = urlrequest.Request(
        endpoint,
        headers={
            "Accept": "application/json",
            "User-Agent": "MovieRecommendationSystem/1.0",
            **({"Authorization": f"Bearer {tmdb_read_token}"} if tmdb_read_token else {}),
        },
        method="GET",
    )

    try:
        with urlrequest.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
            results = payload.get("results", [])
            movies = []
            for item in results[:limit]:
                title = item.get("title") or item.get("name") or "Unknown"
                release = item.get("release_date") or "Unknown"
                rating = item.get("vote_average")
                poster_path = item.get("poster_path")
                movies.append(
                    {
                        "title": title,
                        "release_date": release,
                        "rating": f"{float(rating):.1f}/10" if rating is not None else "N/A",
                        "overview": item.get("overview", ""),
                        "poster_url": f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None,
                    }
                )

            _TRENDING_CACHE["timestamp"] = now
            _TRENDING_CACHE["movies"] = movies
            return movies
    except (urlerror.URLError, TimeoutError, ConnectionResetError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(f"Trending fetch failed: {exc}")
        # One quick retry on transient network failures
        try:
            with urlrequest.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
                results = payload.get("results", [])
                movies = []
                for item in results[:limit]:
                    title = item.get("title") or item.get("name") or "Unknown"
                    release = item.get("release_date") or "Unknown"
                    rating = item.get("vote_average")
                    poster_path = item.get("poster_path")
                    movies.append(
                        {
                            "title": title,
                            "release_date": release,
                            "rating": f"{float(rating):.1f}/10" if rating is not None else "N/A",
                            "overview": item.get("overview", ""),
                            "poster_url": f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None,
                        }
                    )
                _TRENDING_CACHE["timestamp"] = time.time()
                _TRENDING_CACHE["movies"] = movies
                return movies
        except Exception:
            return cached_movies[:limit]


class MovieRecommender:
    """Integrated recommender system matching training/infer.py logic"""
    
    def __init__(self, model_dir='models', progress_callback=None):
        """Initialize with trained model directory"""
        self.model_dir = Path(model_dir).resolve()
        self.metadata = None
        self.similarity_matrix = None
        self.title_to_idx = None
        self.config = None
        self._load_models(progress_callback)
    
    def _load_models(self, progress_callback=None):
        """Load all model artifacts with progress tracking"""
        global _MODEL_LOAD_PROGRESS
        logger.info(f"Loading models from {self.model_dir}...")
        
        # Load metadata (25%)
        if progress_callback:
            progress_callback(10)
        self.metadata = pd.read_parquet(self.model_dir / 'movie_metadata.parquet')
        if progress_callback:
            progress_callback(25)
        
        # Load similarity matrix (sparse .npz or dense .npy) (50%)
        if progress_callback:
            progress_callback(40)
        npz_path = self.model_dir / 'similarity_matrix.npz'
        npy_path = self.model_dir / 'similarity_matrix.npy'
        try:
            if npz_path.exists():
                self.similarity_matrix = load_npz(npz_path)
            elif npy_path.exists():
                self.similarity_matrix = np.load(npy_path)
            else:
                raise FileNotFoundError(
                    f"Neither similarity_matrix.npz nor similarity_matrix.npy found in {self.model_dir}"
                )
        except (FileNotFoundError, OSError) as e:
            # Retry with absolute path if model_dir was relative
            npz_path = Path(npz_path).resolve()
            npy_path = Path(npy_path).resolve()
            if npz_path.exists():
                self.similarity_matrix = load_npz(npz_path)
            elif npy_path.exists():
                self.similarity_matrix = np.load(npy_path)
            else:
                raise FileNotFoundError(
                    f"Similarity matrix not found in {self.model_dir}. Looked for .npz and .npy. {e}"
                ) from e
        if progress_callback:
            progress_callback(65)
        
        # Load title mapping (75%)
        with open(self.model_dir / 'title_to_idx.json', 'r') as f:
            self.title_to_idx = json.load(f)
        if progress_callback:
            progress_callback(80)
        
        # Load config (100%)
        with open(self.model_dir / 'config.json', 'r') as f:
            self.config = json.load(f)
        if progress_callback:
            progress_callback(100)
        
        logger.info(f"Loaded {self.config['n_movies']:,} movies successfully")
    
    def find_movie(self, title: str) -> Optional[str]:
        """Find closest matching movie title"""
        matches = get_close_matches(title, self.title_to_idx.keys(), n=1, cutoff=0.6)
        return matches[0] if matches else None
    
    def search_movies(self, query: str, n: int = 20) -> List[str]:
        """Search movies by partial title"""
        query_lower = query.lower()
        return [title for title in self.title_to_idx.keys() 
                if query_lower in title.lower()][:n]
    
    def get_recommendations(
        self,
        movie_title: str,
        n: int = 15,
        min_rating: float = None
    ) -> Dict:
        """Get movie recommendations with optional filtering"""
        matched_title = self.find_movie(movie_title)
        if not matched_title:
            return {'error': f"Movie '{movie_title}' not found", 'suggestions': self.search_movies(movie_title, 5)}
        
        movie_idx = self.title_to_idx[matched_title]
        source_movie = self.metadata.iloc[movie_idx]
        
        # Get similarity scores (dense or sparse)
        if hasattr(self.similarity_matrix, "getrow"):
            row = self.similarity_matrix.getrow(movie_idx)
            idx_score_pairs = list(zip(row.indices.tolist(), row.data.tolist()))
            idx_score_pairs.sort(key=lambda x: x[1], reverse=True)
            idx_score_pairs = [(i, s) for i, s in idx_score_pairs if i != movie_idx]
        else:
            scores = np.asarray(self.similarity_matrix[movie_idx]).ravel()
            idx_score_pairs = [(i, float(scores[i])) for i in range(scores.shape[0]) if i != movie_idx]
            idx_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for idx, score in idx_score_pairs:
            if len(recommendations) >= n:
                break
            
            movie = self.metadata.iloc[idx]
            
            # Rating filter
            if min_rating and movie['vote_average'] < min_rating:
                continue
            
            recommendations.append({
                'title': movie['title'],
                'release_date': movie['release_date'] if pd.notna(movie['release_date']) else 'Unknown',
                'production': movie['primary_company'] if pd.notna(movie['primary_company']) else 'Unknown',
                'genres': ', '.join(movie['genres'][:3]) if isinstance(movie['genres'], list) else 'N/A',
                'rating': f"{movie['vote_average']:.1f}/10" if pd.notna(movie['vote_average']) else 'N/A',
                'votes': f"{movie['vote_count']:,}" if pd.notna(movie['vote_count']) else 'N/A',
                'similarity_score': f"{score:.3f}",
                'imdb_id': movie['imdb_id'] if pd.notna(movie['imdb_id']) else None,
                'poster_url': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if pd.notna(movie['poster_path']) else None,
                'google_link': f"https://www.google.com/search?q={'+'.join(movie['title'].split())}+movie",
                'imdb_link': f"https://www.imdb.com/title/{movie['imdb_id']}" if pd.notna(movie['imdb_id']) else None
            })
        
        return {
            'query_movie': matched_title,
            'source_movie': {
                'production': source_movie['primary_company'] if pd.notna(source_movie['primary_company']) else 'Unknown',
                'rating': f"{source_movie['vote_average']:.1f}/10" if pd.notna(source_movie['vote_average']) else 'N/A',
                'genres': ', '.join(source_movie['genres'][:3]) if isinstance(source_movie['genres'], list) else 'N/A'
            },
            'recommendations': recommendations
        }


def _load_model_in_background():
    """Load model in background thread"""
    global _RECOMMENDER, _MODEL_LOADING, _MODEL_LOAD_PROGRESS, _LOAD_ERROR
    
    _MODEL_LOADING = True
    _MODEL_LOAD_PROGRESS = 0
    _LOAD_ERROR = None
    
    # Check for model directory (configurable via settings or environment)
    model_dir = getattr(settings, 'MODEL_DIR', os.environ.get('MODEL_DIR', 'models'))
    model_dir = Path(model_dir).resolve()
    # Fallback to project-level models/ then static if training models don't exist
    if not model_dir.exists():
        fallback = Path(settings.BASE_DIR) / 'models'
        if fallback.exists():
            model_dir = fallback
            logger.info(f"Using fallback model directory: {model_dir}")
        else:
            model_dir = Path(settings.BASE_DIR) / 'static'
            logger.warning(f"Model directory not found, using static directory")
    
    try:
        def progress_callback(progress):
            global _MODEL_LOAD_PROGRESS
            _MODEL_LOAD_PROGRESS = progress
            logger.info(f"Model loading progress: {progress}%")
        
        _RECOMMENDER = MovieRecommender(model_dir, progress_callback)
        _MODEL_LOADING = False
        _MODEL_LOAD_PROGRESS = 100
        logger.info("Model loaded successfully")
    except Exception as e:
        _MODEL_LOADING = False
        _LOAD_ERROR = str(e)
        logger.error(f"Failed to load recommender: {e}")


def _start_model_loading():
    """Start model loading in background if not already started"""
    global _LOADING_THREAD, _RECOMMENDER, _MODEL_LOADING
    
    if _RECOMMENDER is None and not _MODEL_LOADING:
        if _LOADING_THREAD is None or not _LOADING_THREAD.is_alive():
            logger.info("Starting model loading in background...")
            _LOADING_THREAD = threading.Thread(target=_load_model_in_background, daemon=True)
            _LOADING_THREAD.start()


def _get_recommender():
    """Get or initialize the recommender singleton"""
    global _RECOMMENDER, _LOAD_ERROR
    
    if _RECOMMENDER is None:
        _start_model_loading()
        if _LOAD_ERROR:
            raise Exception(_LOAD_ERROR)
        return None
    
    return _RECOMMENDER


@require_http_methods(["GET", "POST"])
def main(request):
    """
    Main view for movie recommendation system.
    GET: Display search interface
    POST: Process search and display recommendations
    """
    # Start loading model if not already loading/loaded
    _start_model_loading()
    
    recommender = _get_recommender()
    
    # If model is still loading, show the page with loading state
    if recommender is None:
        if request.method == 'GET':
            context = {
                'all_movie_names': [],
                'total_movies': 0,
            }
            context.update(_build_frontend_settings())
            return render(request, 'recommender/index.html', context)
        else:
            # For POST requests, return error if model not ready
            context = {
                'all_movie_names': [],
                'total_movies': 0,
                'error_message': 'Model is still loading. Please wait a moment and try again.',
            }
            context.update(_build_frontend_settings())
            return render(request, 'recommender/index.html', context)
    
    # Model is loaded, proceed normally
    titles_list = list(recommender.title_to_idx.keys())
    
    if request.method == 'GET':
        context = {
            'all_movie_names': titles_list,
            'total_movies': len(titles_list),
        }
        context.update(_build_frontend_settings())
        return render(
            request,
            'recommender/index.html',
            context
        )
    
    # POST request - process search
    movie_name = request.POST.get('movie_name', '').strip()
    
    if not movie_name:
        context = {
            'all_movie_names': titles_list,
            'total_movies': len(titles_list),
            'error_message': 'Please enter a movie name.',
        }
        context.update(_build_frontend_settings())
        return render(
            request,
            'recommender/index.html',
            context
        )
    
    # Get recommendations
    result = recommender.get_recommendations(movie_name, n=15)
    
    if 'error' in result:
        context = {
            'all_movie_names': titles_list,
            'total_movies': len(titles_list),
            'input_movie_name': movie_name,
            'error_message': result['error'],
            'suggestions': result.get('suggestions', [])
        }
        context.update(_build_frontend_settings())
        return render(
            request,
            'recommender/index.html',
            context
        )
    context = {
        'all_movie_names': titles_list,
        'input_movie_name': result['query_movie'],
        'source_movie': result['source_movie'],
        'recommended_movies': result['recommendations'],
        'total_recommendations': len(result['recommendations']),
    }
    context.update(_build_frontend_settings())
    return render(
        request,
        'recommender/result.html',
        context
    )


@require_http_methods(["GET"])
def auth_page(request):
    """Render Firebase authentication page."""
    context = _build_frontend_settings()
    return render(request, "recommender/auth.html", context)


@require_http_methods(["GET"])
def watchlist_page(request):
    """Render watchlist page shell; data loads via API."""
    context = _build_frontend_settings()
    return render(request, "recommender/watchlist.html", context)


@require_http_methods(["GET"])
def search_movies(request):
    """API endpoint for searching movies (autocomplete)"""
    query = request.GET.get('q', '').strip()
    
    if len(query) < 2:
        return JsonResponse({'movies': [], 'count': 0})
    
    try:
        recommender = _get_recommender()
        
        if recommender is None:
            return JsonResponse({'movies': [], 'count': 0, 'loading': True})
        
        matching_movies = recommender.search_movies(query, n=20)
        
        return JsonResponse({
            'movies': matching_movies,
            'count': len(matching_movies)
        })
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return JsonResponse({'error': 'Search failed'}, status=500)


@require_http_methods(["GET"])
def model_status(request):
    """API endpoint to check model loading status"""
    global _RECOMMENDER, _MODEL_LOADING, _MODEL_LOAD_PROGRESS, _LOAD_ERROR
    
    # Start loading if not already started
    _start_model_loading()
    
    if _LOAD_ERROR:
        return JsonResponse({
            'loaded': False,
            'progress': 0,
            'status': 'error',
            'error': _LOAD_ERROR
        })
    elif _RECOMMENDER is not None:
        return JsonResponse({
            'loaded': True,
            'progress': 100,
            'status': 'ready'
        })
    elif _MODEL_LOADING:
        return JsonResponse({
            'loaded': False,
            'progress': _MODEL_LOAD_PROGRESS,
            'status': 'loading'
        })
    else:
        return JsonResponse({
            'loaded': False,
            'progress': 0,
            'status': 'initializing'
        })


@require_http_methods(["GET"])
def trending_movies(request):
    """API endpoint for real-time trending movies."""
    movies = _fetch_trending_movies(limit=10)
    has_tmdb = bool(
        os.environ.get("TMDB_API_KEY", "").strip()
        or os.environ.get("TMDB_READ_ACCESS_TOKEN", "").strip()
    )
    return JsonResponse({
        "movies": movies,
        "count": len(movies),
        "source": "tmdb" if has_tmdb else "cache",
    })


@csrf_exempt
@require_http_methods(["GET", "POST", "DELETE"])
def watchlist_api(request):
    """CRUD API for Firebase user watchlist stored in Supabase."""
    if request.method == "GET":
        firebase_uid = request.GET.get("uid", "").strip()
        if not firebase_uid:
            return JsonResponse(
                {"success": False, "message": "Missing uid", "items": []},
                status=400,
            )

        items, error = list_watchlist(firebase_uid)
        if error:
            return JsonResponse(
                {"success": False, "message": error, "items": []},
                status=503,
            )
        return JsonResponse({"success": True, "message": "ok", "items": items})

    payload = _json_payload(request)
    firebase_uid = str(payload.get("uid", "")).strip()
    title = str(payload.get("title", "")).strip()

    if not firebase_uid or not title:
        return JsonResponse(
            {"success": False, "message": "uid and title are required", "items": []},
            status=400,
        )

    if request.method == "POST":
        save_error = add_watchlist_item(
            firebase_uid=firebase_uid,
            title=title,
            poster_url=payload.get("poster_url"),
            release_date=payload.get("release_date"),
            rating=payload.get("rating"),
        )
        if save_error:
            return JsonResponse(
                {"success": False, "message": save_error, "items": []},
                status=503,
            )

        items, list_error = list_watchlist(firebase_uid)
        if list_error:
            return JsonResponse(
                {"success": False, "message": list_error, "items": []},
                status=503,
            )
        return JsonResponse({"success": True, "message": "Added to watchlist", "items": items})

    delete_error = remove_watchlist_item(firebase_uid=firebase_uid, title=title)
    if delete_error:
        return JsonResponse(
            {"success": False, "message": delete_error, "items": []},
            status=503,
        )
    items, list_error = list_watchlist(firebase_uid)
    if list_error:
        return JsonResponse(
            {"success": False, "message": list_error, "items": []},
            status=503,
        )
    return JsonResponse({"success": True, "message": "Removed from watchlist", "items": items})


@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint for monitoring"""
    try:
        recommender = _get_recommender()
        return JsonResponse({
            'status': 'healthy',
            'movies_loaded': recommender.config['n_movies'],
            'model_dir': str(recommender.model_dir),
            'model_loaded': True
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JsonResponse({
            'status': 'unhealthy',
            'error': str(e)
        }, status=503)
