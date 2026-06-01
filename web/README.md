# PulmoScan AI — Web UI

Minimal monochrome **Next.js** (App Router) frontend. Upload a chest CT image,
get the model's classification with per-class probabilities.

## Run (end to end)

From the **repo root**:

```bash
make up     # builds + starts the API in Docker on :8000 (waits until healthy)
make web    # installs deps (first run) + starts this UI on :3000
```

Open **http://localhost:3000**, drop in a scan (try one from `Data/test/`), and
hit **Analyze**.

## Standalone

```bash
npm install
npm run dev          # http://localhost:3000
```

## Config

Copy `.env.local.example` → `.env.local` to override defaults:

| Var                   | Default                 | Purpose                                   |
| --------------------- | ----------------------- | ----------------------------------------- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | base URL of the PulmoScan API             |
| `NEXT_PUBLIC_API_KEY` | *(empty)*               | sent as `X-API-Key` if the API requires it |

The browser calls the API directly; the API's default `CORS_ORIGINS` already
allows `http://localhost:3000`.
