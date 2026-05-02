# Deploy on Render (GitHub)

## 1. Push this repo to GitHub

1. Create a **new empty repository** on GitHub (no README if you want a clean first push).
2. Point `origin` at your repo and push:

```bash
cd /path/to/Movie-Recommendation-System-master
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
git branch -M main
git push -u origin main
```

Use SSH instead of HTTPS if you prefer:

```bash
git remote add origin git@github.com:<YOUR_USERNAME>/<YOUR_REPO>.git
```

## 2. Create Render Web Service

1. [Render Dashboard](https://dashboard.render.com) → **New +** → **Blueprint** (if using `render.yaml`) or **Web Service**.
2. Connect the GitHub repo and branch `main`.
3. Render reads [`render.yaml`](/render.yaml) if you use Blueprint deploy.

## 3. Set environment variables on Render

In the service → **Environment**, set (sync these manually if not using Blueprint):

| Variable | Notes |
|----------|--------|
| `ALLOWED_HOSTS` | Your Render hostname, e.g. `movie-recommendation-system.onrender.com` |
| `SUPABASE_URL` | Project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role (server only) |
| `TMDB_API_KEY` | v3 API key |
| `TMDB_READ_ACCESS_TOKEN` | v4 read token (recommended for trending) |
| `FIREBASE_*` | All keys from Firebase console web app |

`SECRET_KEY` and `DEBUG=False` are already in `render.yaml`; `MODEL_DIR` defaults to `./training/models` (fits GitHub size limits).

## 4. Verify

- `https://<your-service>.onrender.com/`
- `https://<your-service>.onrender.com/api/trending/`
- `https://<your-service>.onrender.com/api/health/`

First deploy may take several minutes while dependencies install and the ML model loads.
