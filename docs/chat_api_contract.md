# Chat API Contract

## Authentication

All chat API requests require `Authorization: Bearer <token>`.

## Endpoints

### `POST /chat/start`

Starts a brand-new chat session and queues the first assistant response stream.

Content type: `multipart/form-data`

Fields:
- `message` (string, required)
- `task_id` (integer, optional)
- `image` (file, optional, image only)

Response:

```json
{
  "success": true,
  "message": "Chat session created",
  "data": {
    "sessionId": "uuid",
    "streamId": "uuid",
    "streamUrl": "/chat/uuid"
  }
}
```

### `POST /chat/{session_id}/messages`

Adds a follow-up user message to an existing session and returns a new stream.

Content type: `multipart/form-data`

Fields:
- `message` (string, required)
- `image` (file, optional, image only)

Response:

```json
{
  "success": true,
  "message": "Chat message queued",
  "data": {
    "sessionId": "uuid",
    "streamId": "uuid",
    "streamUrl": "/chat/uuid"
  }
}
```

### `GET /chat/history`

Returns the current user's chat session list.

Query params:
- `start` (integer, optional, default `0`): zero-based offset into the latest-first session list
- `limit` (integer, optional, default `20`, max `100`): number of sessions to return

Response shape:

```json
{
  "success": true,
  "message": "Chat history fetched",
  "data": {
    "sessions": [
      {
        "id": "uuid",
        "title": "How do I improve my serve?",
        "taskId": 42,
        "summary": "Short summary",
        "lastStreamId": "uuid",
        "lastMessagePreview": "Latest message preview",
        "lastAttachmentImageUrl": "/uploads/chat_attachments/serve.png",
        "updatedAt": "2026-04-27T00:00:00Z"
      }
    ],
    "contextSummary": "Readable user memory summary",
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

Ordering:
- sessions are selected latest-first by `updatedAt`

### `GET /chat/history/{session_id}`

Returns transcript data for one existing session.

Query params:
- `start` (integer, optional, default `0`): zero-based offset from the latest side of the chat history
- `limit` (integer, optional, default `10`, max `100`): number of messages to return

Message window behavior:
- `start=0&limit=10` returns the latest 10 chat items in the session
- `start=10&limit=10` returns the next older 10 chat items
- selected messages are returned in chronological order within the window so the UI can render them top-to-bottom without re-sorting

Response shape:

```json
{
  "success": true,
  "message": "Chat session fetched",
  "data": {
    "session": {
      "id": "uuid",
      "title": "Chat title",
      "taskId": 42,
      "summary": "Short summary",
      "lastStreamId": "uuid"
    },
    "messages": [
      {
        "id": 1,
        "role": "user",
        "content": "Question text",
        "metadata": {
          "taskId": 42
        },
        "attachments": [
          {
            "id": 1,
            "type": "image",
            "filename": "serve.png",
            "mimeType": "image/png",
            "fileSize": 12345,
            "url": "/uploads/chat_attachments/serve.png",
            "viewUrl": "/uploads/chat_attachments/serve.png",
            "downloadUrl": "/uploads/chat_attachments/serve.png"
          }
        ],
        "createdAt": "2026-04-27T00:00:00Z"
      },
      {
        "id": 2,
        "role": "assistant",
        "content": "Answer text",
        "metadata": {
          "sources": [
            {
              "type": "document",
              "title": "ITF Rules 2026",
              "governingBody": "ITF",
              "competition": "Grand Slam",
              "seasonYear": 2026,
              "pageStart": 4,
              "pageEnd": 5,
              "pageRange": "4-5",
              "lineStart": 1,
              "lineEnd": 18,
              "viewUrl": "/uploads/knowledge_documents/itf-rules-2026.pdf",
              "downloadUrl": "/uploads/knowledge_documents/itf-rules-2026.pdf"
            },
            {
              "type": "user_memory",
              "summary": "User has been working on serve placement and movement recovery.",
              "source": "chat_session"
            }
          ]
        },
        "attachments": [],
        "createdAt": "2026-04-27T00:00:10Z"
      }
    ],
    "pagination": {
      "start": 0,
      "limit": 10,
      "returned": 2,
      "total": 2,
      "hasMore": false
    }
  }
}
```

### `GET /chat/{stream_id}`

Streams assistant output as `text/event-stream`.

SSE events:
- `started`
- `retrieval`
- `delta`
- `completed`
- `error`

Example event stream:

```text
event: started
data: {"streamId":"uuid","sessionId":"uuid"}

event: retrieval
data: {"sources":[{"type":"document","title":"ITF Rules 2026","pageRange":"4-5","viewUrl":"/uploads/knowledge_documents/itf-rules-2026.pdf"}]}

event: delta
data: {"content":"Start of answer"}

event: completed
data: {"messageId":2}
```

## Access Rules

- `task_id` is optional on `POST /chat/start`.
- When `task_id` is present, only the task owner, tagged opponent, or admin may access the session and its streams.
- History endpoints return only the authenticated user's own sessions.

## Retrieval Source Rules

- document citations do not return numeric document ids
- document citations return document title, page range, line range, and view/download URLs
- user memory citations are deduplicated before being returned
- chat attachment payloads include direct image URLs for rendering in the app

## App Flow

1. Call `POST /chat/start`.
2. Connect to the returned `streamUrl`.
3. Append future turns with `POST /chat/{session_id}/messages`.
4. Reload history with `GET /chat/history`.
5. Reopen a session with `GET /chat/history/{session_id}` and reconnect using the latest stream id if needed.
