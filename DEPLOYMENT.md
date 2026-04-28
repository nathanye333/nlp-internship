# Deploying the Real Estate NLP demo

The Week 11 product integration ships as two processes -- a FastAPI service
(`scripts.production_api:app`) and a Streamlit UI (`scripts/product_demo.py`)
-- backed by the same Docker image with a `RUN_MODE` switch.

This guide focuses on **Render** (recommended for the demo) and adds short
notes for **Railway** as a drop-in alternative. Heroku no longer offers a
free tier; if you need it, scaling instructions mirror Render's.

## TL;DR

```bash
git push origin main           # blueprint auto-deploys on push
```

After the first deploy the Streamlit UI will be at
`https://nlp-ui.onrender.com`, talking to the API at
`https://nlp-api.onrender.com`.

## Architecture recap

| Service | What runs                                              | Port     |
| ------- | ------------------------------------------------------ | -------- |
| nlp-api | `uvicorn scripts.production_api:app`                   | `$PORT`  |
| nlp-ui  | `streamlit run scripts/product_demo.py`                | `$PORT`  |

Both services share the same Docker image. `RUN_MODE=api` (default) starts
uvicorn; `RUN_MODE=ui` starts Streamlit. The UI reads `API_BASE_URL` to
locate the API.

## Render

### 1. One-time prep

1. Push the repo to a GitHub or GitLab account that Render can read.
2. Sign in to <https://dashboard.render.com> with that account.
3. Make sure `data/models/listings_semantic.faiss` and
   `data/models/listings_semantic_meta.json` are committed (or pulled in via
   a build step). Without those artefacts the API starts but `/search`
   returns 503 until you upload them or fall back to keyword-only mode (see
   "Free-tier caveats" below).

### 2. Launch the blueprint

1. In the Render dashboard click **New + → Blueprint**.
2. Pick this repo. Render auto-detects [`render.yaml`](render.yaml) and
   provisions two web services (`nlp-api`, `nlp-ui`).
3. Review the env vars. The blueprint wires `nlp-ui`'s `API_BASE_URL` to the
   internal hostport of `nlp-api` automatically.
4. Click **Apply**. The first build takes ~6-10 minutes (most of it is
   downloading `faiss-cpu` + `sentence-transformers`).

### 3. Verify

```bash
curl https://nlp-api.onrender.com/health
# {"status":"ok","version":"1.0.0","semantic_search_ready":false}

curl -X POST https://nlp-api.onrender.com/search \
  -H 'content-type: application/json' \
  -d '{"query":"3 bed 2 bath under 700k in Irvine","mode":"hybrid","top_k":3}'
```

Then open `https://nlp-ui.onrender.com` and run a search. The **Metrics**
tab should show your traffic.

### 4. Free-tier caveats (read this!)

- Render's free-tier services sleep after 15 minutes of inactivity. The
  first request after a sleep takes 30-60 seconds while the container
  spins back up.
- **The semantic searcher needs more than 512 MB of RAM.** Sentence-
  transformer + FAISS + the listings index push memory close to 1 GB on
  first warmup; on the free 512 MB plan the container OOMs. Two options:
  1. Upgrade `nlp-api` to **Starter ($7/mo)** -- this is what `render.yaml`
     selects by default. The UI can stay on Starter or be moved to Streamlit
     Community Cloud at no cost.
  2. **Disable semantic search and ship a keyword-only demo** by removing
     the FAISS artefacts from the image. The API catches the missing index
     and gracefully serves only `/search?mode=keyword` -- the Compare tab
     will mark `semantic` and `hybrid` as unavailable, which is itself a
     useful demo of graceful degradation.
- The feedback log (`data/processed/demo_event_log.jsonl`) lives on
  Render's ephemeral filesystem and resets when the container restarts. For
  persistent feedback, mount a Render Disk or swap the
  `append_feedback_event` call for an external store (S3 / Postgres /
  Supabase).

### 5. Updating the deploy

`autoDeploy: true` is set in `render.yaml`, so every push to the tracked
branch triggers a rebuild on both services. Use **Manual Deploy** in the
dashboard if you need to redeploy without a push (e.g. after rotating an
env var).

## Railway (alternative)

Railway has a small free trial, after which it charges by usage. The same
image works there too:

1. `railway init` then `railway up` from the repo root.
2. Create two services from the same repo: set `RUN_MODE=api` on the first
   and `RUN_MODE=ui, API_BASE_URL=https://<api-service>.up.railway.app` on
   the second.
3. Set the start command to `bash -lc 'echo $PORT && exec [...]'` if Railway
   doesn't auto-detect `EXPOSE`. The Dockerfile already honours `$PORT`.

## Local Docker parity

`docker compose up` brings up MySQL + the API + the UI on the same network
(`api` is reachable as `http://api:8000` from the UI container). This is
the recommended local smoke test before deploying.

```bash
docker compose up --build
open http://localhost:8501       # Streamlit UI
open http://localhost:8000/docs  # FastAPI Swagger
```

## Troubleshooting

| Symptom                                       | Fix |
| --------------------------------------------- | --- |
| UI shows "API error: ConnectError"            | Confirm `API_BASE_URL` env var is set on `nlp-ui` and the API service is awake (visit `/health`). |
| `/search` returns 503 "Semantic searcher unavailable" | Index artefacts missing -- either rebuild them with `scripts/semantic_searcher.py --build` or fall back to `mode=keyword`. |
| Containers killed with exit code 137 (OOM)    | Bump the API service plan from Free to Starter, or remove the FAISS index to disable semantic mode. |
| Rate-limit 429s during demo                   | Default is 10 req/s per IP. Increase the `default_limits=` in `production_api.py` or run multiple Render replicas behind their load balancer. |
