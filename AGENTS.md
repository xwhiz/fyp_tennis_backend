# Repo Notes

## Source Of Truth
- Use `pyproject.toml` + `uv.lock` for dependencies and test tooling. `requirements.txt` is stale/partial and omits much of the real runtime stack.
- No repo CI workflow, lint config, formatter config, or typecheck config is checked in. Verified automation source of truth is `pytest` plus the Docker commands below.

## Runtime Entrypoints
- FastAPI app entrypoint is `src.main:app`.
- `docker compose up --build` is the main full-stack dev path. The API container runs `uv run alembic upgrade head && uv run uvicorn src.main:app --host 0.0.0.0 --port 7000 --workers 4`.
- Celery worker entrypoint: `uv run celery -A src.celery.worker worker --loglevel=info --concurrency=1`.
- Flower entrypoint: `uv run celery --broker=amqp://guest:guest@rabbitmq:5672// flower --port=5555`.

## Env And DB
- `src.config.Settings` loads the root `.env`. Alembic also calls `load_dotenv()` and hard-fails if `DATABASE_URL` is unset.
- Defaults and README examples assume Docker hostnames (`postgres`, `rabbitmq`, `redis`). If running outside Docker, override them.
- App startup (`src.main.lifespan`) seeds the admin user and requeues `pending` / `processing` background tasks. Any `TestClient(app)` run triggers that startup logic.

## Tests
- Use `uv run pytest`.
- Fast default suite: `uv run pytest tests -m "not (performance or security)"`.
- Marker runs: `uv run pytest tests -m unit`, `uv run pytest tests -m integration`, `uv run pytest tests -m system`, `uv run pytest tests -m performance`, `uv run pytest tests -m security`.
- Focus one file or case with normal pytest selectors, e.g. `uv run pytest tests/integration/test_api_tasks.py -k process_video`.
- Standard test runs do not need Postgres, Redis, or RabbitMQ. `tests/conftest.py` forces `DATABASE_URL=sqlite:///:memory:` before `src.*` imports and mocks Celery where needed.
- If a test needs different env/config, set env before importing any `src.*` module. `settings` and the SQLAlchemy engine are created at import time.

## Code Layout
- `src/api/`: HTTP routes.
- `src/celery/worker.py`: background video-processing task and model lazy-loading.
- `src/core/`: ML / CV pipeline.
- `src/services/`: auth and stats shaping.
- `src/db/`: engine, sessions, persistence helpers.

## API Pagination Rules
- New endpoints that return potentially unbounded item lists should support pagination from day one.
- Use explicit query params in the API contract for list pagination and document their defaults, ordering, and response metadata.
- If pagination is intentionally omitted for a list-style endpoint, document why the endpoint is intentionally bounded or why additional calls would be unnecessary.

## ML / File Gotchas
- Worker/model code expects weight files at repo-relative paths: `src/track_net_weights.pt`, `src/model_tennis_court_det.pt`, `src/ctb_regr_bounce.cbm`, and root `yolo26x.pt`. Run worker/API commands from repo root unless you also update those paths.
- `src.main` creates and mounts `uploads/` and `output/` as static dirs.

## Schema / Migration Gotchas
- Alembic autoloads every module under `src.models` via `pkgutil.iter_modules`, so new model files under that package are seen by migrations without a manual registry.
- Tests are stricter: `tests/conftest.py` manually imports models before `Base.metadata.create_all(engine)`. If you add a model used in tests, import it there or the SQLite test schema will miss the table.
