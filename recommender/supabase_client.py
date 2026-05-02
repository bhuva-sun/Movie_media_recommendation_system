"""Supabase helper functions for per-user watchlists."""

from typing import Dict, List, Optional, Tuple

from django.conf import settings

try:
    from supabase import Client, create_client
except ImportError:  # pragma: no cover
    Client = None
    create_client = None


def _get_client() -> Tuple[Optional["Client"], Optional[str]]:
    """Return Supabase client and error string (if any)."""
    if create_client is None:
        return None, "Supabase dependency is not installed."

    supabase_url = getattr(settings, "SUPABASE_URL", "").strip()
    service_role_key = getattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not supabase_url or not service_role_key:
        return None, "Supabase credentials are missing."

    # Accept either project URL or PostgREST URL.
    if "/rest/v1" in supabase_url:
        supabase_url = supabase_url.split("/rest/v1", 1)[0].rstrip("/")

    try:
        return create_client(supabase_url, service_role_key), None
    except Exception as exc:  # pragma: no cover
        return None, f"Failed to initialize Supabase client: {exc}"


def list_watchlist(firebase_uid: str) -> Tuple[List[Dict], Optional[str]]:
    """Fetch watchlist entries for a Firebase user."""
    client, error = _get_client()
    if error:
        return [], error

    try:
        response = (
            client.table("watchlist_items")
            .select("title,poster_url,release_date,rating,created_at")
            .eq("firebase_uid", firebase_uid)
            .order("created_at", desc=True)
            .execute()
        )
        return response.data or [], None
    except Exception as exc:
        message = str(exc)
        if "403" in message:
            return [], "Supabase access forbidden (403). Use SUPABASE_SERVICE_ROLE_KEY, or disable RLS for watchlist_items."
        return [], f"Failed to fetch watchlist: {exc}"


def add_watchlist_item(
    firebase_uid: str,
    title: str,
    poster_url: Optional[str] = None,
    release_date: Optional[str] = None,
    rating: Optional[str] = None,
) -> Optional[str]:
    """Insert or update a watchlist item for a Firebase user."""
    client, error = _get_client()
    if error:
        return error

    payload = {
        "firebase_uid": firebase_uid,
        "title": title,
        "poster_url": poster_url,
        "release_date": release_date,
        "rating": rating,
    }
    try:
        # Prefer upsert so repeated adds do not create duplicates.
        client.table("watchlist_items").upsert(
            payload, on_conflict="firebase_uid,title"
        ).execute()
        return None
    except Exception as exc:
        message = str(exc)
        if "403" in message:
            return "Supabase access forbidden (403). Use SUPABASE_SERVICE_ROLE_KEY, or disable RLS for watchlist_items."
        return f"Failed to save watchlist item: {exc}"


def remove_watchlist_item(firebase_uid: str, title: str) -> Optional[str]:
    """Delete one watchlist item for a Firebase user."""
    client, error = _get_client()
    if error:
        return error

    try:
        (
            client.table("watchlist_items")
            .delete()
            .eq("firebase_uid", firebase_uid)
            .eq("title", title)
            .execute()
        )
        return None
    except Exception as exc:
        message = str(exc)
        if "403" in message:
            return "Supabase access forbidden (403). Use SUPABASE_SERVICE_ROLE_KEY, or disable RLS for watchlist_items."
        return f"Failed to remove watchlist item: {exc}"
