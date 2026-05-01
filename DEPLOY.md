# Deploy Beacon — zero-Claude, zero-self-host

End-to-end path to a public Streamlit URL with hosted Postgres + Langfuse Cloud
tracing. None of these steps require a new pipeline run, so no Claude calls are
consumed.

## 1. Neon Postgres (free tier)

1. Sign up at [neon.tech](https://neon.tech) — Hobby tier is free.
2. Create a project. Region near you.
3. Settings → Extensions → enable `postgis` and `vector`.
4. Settings → Connection string → copy the pooled connection string (looks like
   `postgresql://USER:PASS@ep-xxx.neon.tech/neondb?sslmode=require`).
5. Run the migration:
   ```bash
   NEON_URL='postgresql://...' ./scripts/migrate_to_neon.sh
   ```

## 2. Langfuse Cloud (free Hobby tier)

1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com).
2. Create a project named `beacon`.
3. Settings → API Keys → create a public + secret key pair.
4. Add to `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```
5. Future pipeline runs auto-trace. (Existing runs in DB do not.)

## 3. Streamlit Cloud deploy

1. Push the repo to a public GitHub repo.
2. Sign in to [streamlit.io/cloud](https://streamlit.io/cloud) with GitHub.
3. New app → repo `abdllahx/beacon`, branch `main`, main file `src/beacon/app.py`.
4. Advanced settings → Secrets → paste content of `.streamlit/secrets.toml.template`
   with real values.
5. Deploy. App URL will be `https://<repo>.streamlit.app`.

## 4. Verify

- Open the deployed URL — sidebar should show latency p50/p95/p99 over the
  cached 50 verification_runs and any cost_events written by past Claude calls.
- HITL feedback buttons write to the `feedback` table on Neon.
- Langfuse dashboard remains empty until a fresh pipeline run executes.

## What you get on the resume

- Public clickable demo URL
- Tracing screenshots from Langfuse for the blog post
- Per-verification cost number from the cost log (median + total)
- HITL feedback as the bridge to a self-improving system
- DSPy as a "production prompt-engineering" buzzword without the Claude-burn of an actual optimization run
