---
name: Fix test and syntax errors
overview: Fix the SyntaxError in process_video.py that blocks pytest, and optionally avoid loading the full app (and torch) when running unit-only tests so the NumPy/torch stack is not required for tests that do not need the API.
todos: []
isProject: false
---

# Fix test run and syntax errors

## 1. SyntaxError in process_video.py (blocking)

**Error:** `SyntaxError: non-default argument follows default argument` at line 150 in [src/core/process_video.py](src/core/process_video.py).

**Cause:** In `get_detections_from_video`, `task_id` has a default (`Optional[int] = None`) but the next parameter `video_path: str` has no default. In Python, all parameters with defaults must come after parameters without defaults.

**Fix:**

- In **get_detections_from_video** (lines 144–153): move `video_path: str` before `task_id: Optional[int] = None` so the signature is:
  - `ball_detector`, `court_detector`, `person_detector`, `bounce_detector`, `**video_path: str`**, `**task_id: Optional[int] = None`**, `cap=None`, `valid_scenes=None`.
- Update the **single call site** in the same file (lines 419–427) to pass `video_path` before `task_id`:
  - Change from: `task_id`, `video_path`, …
  - To: `video_path`, `task_id`, …

The notebook call uses keyword arguments (`task_id=TASK_ID, video_path=VIDEO_PATH`), so it remains valid and needs no change.

---

## 2. Optional: avoid loading app/torch for unit-only runs

**Issue:** [tests/conftest.py](tests/conftest.py) does `from src.main import app` at import time. That pulls in `main` → `stream_infer` → `person_detector` → `torch`/`torchvision`, which can trigger the NumPy 1.x vs 2.x warning and slow down runs for tests that never use the API.

**Optional improvement:** Lazy-import the app only when a fixture that needs it is used.

- Leave `DATABASE_URL` and DB/engine/table setup as-is at top level (so SQLite and tables are ready).
- Do **not** import `app` at top level. Instead, inside the `client` and `client_no_celery` fixtures, do something like:

```python
  from src.main import app
  with TestClient(app) as c:
      yield c


```

- Ensure any other fixtures that depend on `app` (e.g. if something uses the app object) also import it inside the fixture.

Then:

- **Unit tests** that don’t request `client` or `client_no_celery` (e.g. `test_config`, `test_schemas`, `test_core_utils`, `test_direction_change_indices`) will not load `src.main` or torch, avoiding the NumPy/torch stack for those runs.
- **Integration/system/performance/security** tests that use `client` or `client_no_celery` will still load the app and run as they do now.

If you only need tests to run at all, fixing the SyntaxError (section 1) is sufficient; section 2 is an optional improvement for faster and cleaner unit-only test runs.
