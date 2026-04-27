# Pagination And Search API Contract

This document summarizes the pagination and search contract for the endpoints changed in this rollout, plus how the app should consume them.

## Shared Pagination Format

All paginated endpoints in this rollout use:

- `start`: zero-based offset
- `limit`: page size
- `data.pagination`: pagination metadata returned with the list payload

Shared response metadata:

```json
{
  "pagination": {
    "start": 0,
    "limit": 20,
    "returned": 20,
    "total": 87,
    "hasMore": true
  }
}
```

Field meanings:

- `start`: offset requested by the client
- `limit`: page size requested by the client
- `returned`: number of items actually returned in this response
- `total`: total number of matching items
- `hasMore`: whether another page can be requested

## Changed Endpoints

### `GET /all_tasks`

Query params:

- `start` (optional, default `0`)
- `limit` (optional, default `20`, max `100`)

Response shape:

```json
{
  "success": true,
  "data": {
    "tasks": [],
    "pagination": {
      "start": 0,
      "limit": 20,
      "returned": 0,
      "total": 0,
      "hasMore": false
    }
  }
}
```

Behavior:

- returns tasks visible to the authenticated user
- ordered latest-first by `updated_at`, then `id`

### `GET /all_tasks/search`

Query params:

- `q` (required): matched against task `name` and `description`
- `start` (optional, default `0`)
- `limit` (optional, default `20`, max `100`)

Response shape:

```json
{
  "success": true,
  "data": {
    "tasks": [],
    "pagination": {
      "start": 0,
      "limit": 20,
      "returned": 0,
      "total": 0,
      "hasMore": false
    }
  }
}
```

Behavior:

- only searches tasks the authenticated user is allowed to see
- matching is case-insensitive against task `name` and `description`
- results are ordered latest-first by `updated_at`, then `id`

### `GET /users/search`

Query params:

- `q` (required, min length `2`)
- `start` (optional, default `0`)
- `limit` (optional, default `20`, max `50`)

Response shape:

```json
{
  "success": true,
  "message": "Search results",
  "data": {
    "results": [],
    "pagination": {
      "start": 0,
      "limit": 20,
      "returned": 0,
      "total": 0,
      "hasMore": false
    }
  }
}
```

Behavior:

- results are still ranked so exact and prefix at-tag matches appear first
- pagination is applied after ranking

### `GET /chat/history`

Query params:

- `start` (optional, default `0`)
- `limit` (optional, default `20`, max `100`)

Response shape:

```json
{
  "success": true,
  "message": "Chat history fetched",
  "data": {
    "sessions": [],
    "contextSummary": "Readable user memory summary",
    "pagination": {
      "start": 0,
      "limit": 20,
      "returned": 0,
      "total": 0,
      "hasMore": false
    }
  }
}
```

Behavior:

- sessions are returned latest-first by `updatedAt`

### `GET /chat/history/{session_id}`

Query params:

- `start` (optional, default `0`)
- `limit` (optional, default `10`, max `100`)

Response shape:

```json
{
  "success": true,
  "message": "Chat session fetched",
  "data": {
    "session": {},
    "messages": [],
    "pagination": {
      "start": 0,
      "limit": 10,
      "returned": 0,
      "total": 0,
      "hasMore": false
    }
  }
}
```

Behavior:

- `start=0&limit=10` returns the latest 10 chat items
- `start=10&limit=10` returns the next older 10 chat items
- the selected window is returned in chronological order for rendering

## App Implementation Guide

Use the same paging flow for every paginated endpoint:

1. Request the first page with `start=0`.
2. Render the returned list immediately.
3. Store `pagination.start`, `pagination.limit`, `pagination.returned`, `pagination.total`, and `pagination.hasMore`.
4. When loading more, request the next page with `start = currentStart + returned`.
5. Stop requesting more when `hasMore` is `false`.

Recommended app behavior by endpoint:

- task list screen: call `GET /all_tasks?start=0&limit=20`
- task search screen: call `GET /all_tasks/search?q=<term>&start=0&limit=20`
- user search: call `GET /users/search?q=<term>&start=0&limit=20`
- chat history screen: call `GET /chat/history?start=0&limit=20`
- open chat thread: call `GET /chat/history/{session_id}?start=0&limit=10`

For chat threads specifically:

- prepend older messages when requesting later pages from `GET /chat/history/{session_id}`
- do not reverse the returned page, because the backend already returns the selected window in chronological order

