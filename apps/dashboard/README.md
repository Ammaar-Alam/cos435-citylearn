# CityLearn dashboard

Local React dashboard for launching CityLearn runs, watching jobs, reviewing artifacts, and comparing completed results.

The dashboard sits on top of the repo's FastAPI backend. It uses the same runners, configs, and `results/` artifacts as the command-line benchmark flow, so UI runs and script runs share one local result store.

## What it includes

- Run launcher for RBC, centralized PPO/SAC/TD3, and shared PPO/SAC/TD3 `reward_v2`
- Live job monitor with worker logs and preview playback
- Run browser for completed metrics, traces, render media, and exported artifacts
- Side-by-side run comparison
- Artifact import and checkpoint evaluation flow

## Setup

From the repo root:

```bash
make install-benchmark
source .venv/bin/activate
make download-citylearn
make dashboard-install
```

## Development

Start the backend and frontend together:

```bash
source .venv/bin/activate
make ui
```

Open `http://127.0.0.1:5173/dashboard/`.

Run each side separately:

```bash
make dashboard-backend
make dashboard-frontend
```

The backend runs on `http://127.0.0.1:8001`. The frontend dev server proxies API requests through normal browser fetches to the same host paths used by the built app.

## Production-style build

```bash
make dashboard-build
make dashboard-backend
```

After a build, the FastAPI backend serves the dashboard at `/dashboard`.

## Code map

- `src/router.tsx` defines the dashboard routes.
- `src/routes/overview.tsx` launches benchmark jobs and shows the latest run.
- `src/routes/monitor.tsx` follows active jobs, logs, and preview playback.
- `src/routes/runs.tsx` and `src/routes/run-detail.tsx` inspect saved runs.
- `src/routes/compare.tsx` compares completed runs.
- `src/routes/artifacts.tsx` imports and evaluates external artifacts.
- `src/lib/api.ts` contains the backend API calls.

Backend routes and services live under `src/cos435_citylearn/api/`.
