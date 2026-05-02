"""
URL Configuration for Movie Recommendation System
"""
from django.urls import path
from . import views

app_name = 'recommender'

urlpatterns = [
    # Main views
    path('', views.main, name='main'),
    path('auth/', views.auth_page, name='auth_page'),
    path('watchlist/', views.watchlist_page, name='watchlist_page'),
    
    # API endpoints
    path('api/search/', views.search_movies, name='search_movies'),
    path('api/model-status/', views.model_status, name='model_status'),
    path('api/health/', views.health_check, name='health_check'),
    path('api/trending/', views.trending_movies, name='trending_movies'),
    path('api/watchlist/', views.watchlist_api, name='watchlist_api'),
]
