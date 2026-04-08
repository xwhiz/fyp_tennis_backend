# Testing Plan and Test Cases – FYP Tennis Backend

This document describes the comprehensive test suite for the Tennis Analytics Backend, covering **Functional Testing** (Unit, Integration, System) and **Non-Functional Testing** (Performance, Security) as required by the project rubric.

---

## 1. Purpose

- Provide a **comprehensive list of test cases** and a **testing plan** for the system.
- **Functional testing**: Unit, Integration, and System testing.
- **Non-functional testing**: Performance and Security testing.
- Each category is mapped to the endpoints (or modules) under test where applicable.

---

## 2. Test Structure and How to Run

### 2.1 Directory Layout

```
tests/
├── conftest.py              # Shared fixtures (TestClient, DB, Celery mock, sample_task_id)
├── unit/                    # Unit tests (no HTTP, no external services)
│   ├── test_config.py
│   ├── test_core_utils.py
│   ├── test_direction_change_indices.py
│   ├── test_db_utils.py
│   └── test_schemas.py
├── integration/             # Integration tests (API + test DB)
│   ├── test_api_misc.py
│   ├── test_api_tasks.py
│   ├── test_api_stats.py
│   └── test_api_stream.py
├── system/                  # System / smoke / e2e
│   ├── test_smoke.py
│   └── test_e2e_flow.py
├── performance/             # Performance tests
│   └── test_api_performance.py
└── security/                # Security tests
    └── test_security.py
```

### 2.2 Running Tests

| Command | Description |
|--------|--------------|
| `pytest tests/` | Run all tests |
| `pytest tests/ -m unit` | Run only unit tests |
| `pytest tests/ -m integration` | Run only integration tests |
| `pytest tests/ -m system` | Run only system tests |
| `pytest tests/ -m performance` | Run only performance tests |
| `pytest tests/ -m security` | Run only security tests |
| `pytest tests/ -m "not (performance or security)"` | Run functional tests only (faster, suitable for CI) |
| `pytest tests/ --cov=src` | Run all tests with coverage report |

Integration tests use **in-memory SQLite** (via `DATABASE_URL=sqlite:///:memory:` set in `conftest.py`), so no PostgreSQL is required for the test run. Celery is **mocked** in integration/e2e where needed so no broker or worker is required.

---

## 3. Functional Testing

### 3.1 Unit Testing

Unit tests exercise application logic and DB helpers in isolation. **No HTTP endpoints are called.**

| File | What It Tests |
|------|----------------|
| **tests/unit/test_config.py** | `Settings`: presence of expected attributes, `database_url`, `upload_chunk_size`, `port`, `app_name`. |
| **tests/unit/test_core_utils.py** | `classify_serve_type(bounce_x, bounce_y)` (T/body/wide/unknown); `get_slope(values)` (empty, single value, constant, positive/negative slope); `perspective_transform_point(point, matrix)` (None handling, identity matrix). |
| **tests/unit/test_direction_change_indices.py** | `get_direction_change_indices(ball_track, ...)`: empty track, short track, single direction, one direction change, return type. |
| **tests/unit/test_db_utils.py** | `to_float(x)` (None, list of ints, list with None); `update_task_status` / `update_upload_progress` and save helpers (`save_ball_track_in_db`, `save_video_paths_in_db`, `save_thumbnail_in_db`) using the test DB. |
| **tests/unit/test_schemas.py** | Pydantic schemas: `ProcessVideoResponse`, `VideoPathsSchema`, `BallTrackSchema`, `BouncesSchema`, `ThumbnailSchema`, `DirectionChangeIndicesSchema`, `PlayerPositionsSchema` (valid payloads and missing-required-field validation). |

**Endpoints:** None (logic only).

---

### 3.2 Integration Testing

Integration tests hit the **API** using FastAPI’s `TestClient` and the test database (in-memory SQLite). Celery is mocked where necessary.

| Endpoint | File | What Is Tested |
|----------|------|----------------|
| `GET /` | test_api_misc.py | Returns 200. |
| `GET /check-health` | test_api_misc.py | Returns 200 and body `message: OK`, `success: true`. |
| `GET /court_reference` | test_api_misc.py | Returns 200 and some structure. |
| `GET /all_tasks` | test_api_tasks.py | Returns 200, `success` and `data` list. |
| `GET /task_progress/{process_id}` | test_api_tasks.py | 200 with “Process not found” for missing id; 200 with task data for existing task. |
| `POST /process_video` | test_api_tasks.py | 400 for non-video content-type; 400 when `duplicate_task=true` and no `task_id`; 404 for duplicate with invalid `task_id`; 200 and `process_id` for small file upload (Celery mocked). |
| `POST /upload_chunk/{task_id}` | test_api_tasks.py | 404 for invalid task_id; 400 when task is already fully uploaded. |
| `DELETE /delete_task/{task_id}` | test_api_tasks.py | 404 for missing task; 200 and success for existing task. |
| `GET /get_video_paths/{task_id}` | test_api_stats.py | 200 and “not found” message when no data. |
| `GET /get_speed_stats/{task_id}` | test_api_stats.py | 200 when no data. |
| `GET /get_ball_track/{task_id}` | test_api_stats.py | 200 when no data. |
| `GET /get_bounces/{task_id}` | test_api_stats.py | 200 when no data. |
| `GET /get_direction_change_indices/{task_id}` | test_api_stats.py | 200 when no data. |
| `GET /get_player_positions/{task_id}` | test_api_stats.py | 200 when no data. |
| `GET /thumbnail/{task_id}` | test_api_stats.py | 200 when no data. |
| `GET /all-stats/{task_id}` | test_api_stats.py | 200. |
| `GET /serve_stats/{task_id}` | test_api_stats.py | 200 when no data. |
| `GET /player_heatmaps/{task_id}` | test_api_stats.py | 200. |
| `GET /stream/output/{filename}` | test_api_stream.py | 404 for missing file. |
| `GET /stream/uploads/{filename}` | test_api_stream.py | 404 for missing file. |

---

### 3.3 System Testing

| Test | File | Description | Endpoints Used |
|------|------|-------------|----------------|
| **Smoke** | test_smoke.py | App responds: root, health, all_tasks return 200 and expected shape. | `GET /`, `GET /check-health`, `GET /all_tasks` |
| **E2E flow** | test_e2e_flow.py | Create task via `POST /process_video` (Celery mocked), then `GET /task_progress/{id}`, then `GET /get_ball_track/{id}`; responses are well-formed. | `POST /process_video`, `GET /task_progress/{id}`, `GET /get_ball_track/{id}` |

---

## 4. Non-Functional Testing

### 4.1 Performance Testing

| Test | File | What Is Measured | Endpoints |
|------|------|------------------|-----------|
| **Latency** | test_api_performance.py | Response time for key endpoints; assertions that each stays below a threshold (e.g. 2–3 s). | `GET /check-health`, `GET /all_tasks`, `GET /task_progress/{id}`, `GET /get_ball_track/{id}` |

Pass criteria: each request completes within the configured max time (see `HEALTH_MAX_SEC`, `ALL_TASKS_MAX_SEC`, etc. in the test file).

### 4.2 Security Testing

| Test | File | What Is Verified | Endpoints |
|------|------|------------------|-----------|
| **Path traversal** | test_security.py | `GET /stream/output/../../../etc/passwd` and encoded variants, `GET /stream/uploads/...` – must return 404 or 400 and must not return sensitive file content. | `/stream/output/{filename}`, `/stream/uploads/{filename}` |
| **Input validation** | test_security.py | `POST /process_video` with non-video file → 400; invalid/missing task ids on task_progress, delete_task, upload_chunk → 404 (or expected error shape). | `/process_video`, `/task_progress/{process_id}`, `/upload_chunk/{task_id}`, `/delete_task/{task_id}` |

---

## 5. Endpoint Summary by Test Category

| Category | Endpoints |
|----------|-----------|
| **Unit** | None (config, core utils, direction change, db utils, schemas only). |
| **Integration** | `/`, `/check-health`, `/court_reference`, `/all_tasks`, `/task_progress/{process_id}`, `/process_video`, `/upload_chunk/{task_id}`, `/delete_task/{task_id}`, `/get_video_paths/{task_id}`, `/get_speed_stats/{task_id}`, `/get_ball_track/{task_id}`, `/get_bounces/{task_id}`, `/get_direction_change_indices/{task_id}`, `/get_player_positions/{task_id}`, `/thumbnail/{task_id}`, `/all-stats/{task_id}`, `/serve_stats/{task_id}`, `/player_heatmaps/{task_id}`, `/stream/output/{filename}`, `/stream/uploads/{filename}`. |
| **System** | `/`, `/check-health`, `/all_tasks` (smoke); `/process_video`, `/task_progress/{id}`, `/get_ball_track/{id}` (e2e). |
| **Performance** | `/check-health`, `/all_tasks`, `/task_progress/{id}`, `/get_ball_track/{id}`. |
| **Security** | `/stream/output/{filename}`, `/stream/uploads/{filename}`, `/process_video`, `/task_progress/{process_id}`, `/upload_chunk/{task_id}`, `/delete_task/{task_id}`. |

---

## 6. CI and Optional Runs

- **Default / CI:** Run functional tests only:  
  `pytest tests/ -m "not (performance or security)"`  
  to keep runs fast and avoid flakiness from timing.
- **Full run (including non-functional):**  
  `pytest tests/`
- **Coverage:**  
  `pytest tests/ --cov=src --cov-report=term-missing`

All tests use the test configuration from `tests/conftest.py` (in-memory SQLite and mocked Celery where applicable), so no external database or message broker is required for the standard test suite.
