# Tasks API Contract

## Authentication

All task API requests require `Authorization: Bearer <token>`.

## Pagination Rules

`GET /all_tasks` and `GET /all_tasks/search` use offset-style pagination:

- `start` (integer, optional, default `0`): zero-based offset into the latest-first task list
- `limit` (integer, optional, default `20`, max `100`): number of tasks to return

Ordering:

- tasks are returned latest-first by `updated_at`

## `GET /all_tasks`

Returns the authenticated user's visible tasks. Admins see all tasks; non-admins see tasks they own or where they are the tagged opponent.

Response shape:

```json
{
  "success": true,
  "data": {
    "tasks": [
      {
        "id": "42",
        "name": "Practice match",
        "status": "completed",
        "description": "Processing complete",
        "created_at": "2026-04-27T00:00:00Z",
        "updated_at": "2026-04-27T00:10:00Z",
        "total_upload_size": 123456,
        "uploaded_size": 123456,
        "is_uploaded_fully": true,
        "progress": 100.0,
        "opponent": {
          "id": "uuid",
          "atTag": "@rival",
          "firstName": "Rival",
          "lastName": "Player"
        }
      }
    ],
    "pagination": {
      "start": 0,
      "limit": 20,
      "returned": 1,
      "total": 1,
      "hasMore": false
    }
  }
}
```

Pagination behavior:

- `GET /all_tasks?start=0&limit=20` returns the first page of the latest tasks
- `GET /all_tasks?start=20&limit=20` returns the next page
- if `pagination.hasMore` is `true`, the client can request the next page using `start += returned`

## `GET /all_tasks/search`

Searches the authenticated user's visible tasks and returns the same paginated task payload shape as `GET /all_tasks`.

Query params:

- `q` (string, required, min length `1`): search term matched against task `name` and `description`
- `start` (integer, optional, default `0`)
- `limit` (integer, optional, default `20`, max `100`)

Response shape:

```json
{
  "success": true,
  "data": {
    "tasks": [
      {
        "id": "42",
        "name": "Serve practice",
        "status": "completed",
        "description": "Morning serve session",
        "created_at": "2026-04-27T00:00:00Z",
        "updated_at": "2026-04-27T00:10:00Z",
        "total_upload_size": 123456,
        "uploaded_size": 123456,
        "is_uploaded_fully": true,
        "progress": 100.0,
        "opponent": null
      }
    ],
    "pagination": {
      "start": 0,
      "limit": 20,
      "returned": 1,
      "total": 3,
      "hasMore": true
    }
  }
}
```

Search behavior:

- results are filtered to tasks visible to the authenticated user
- matching is case-insensitive against task `name` and `description`
- results are still ordered latest-first by `updated_at`
