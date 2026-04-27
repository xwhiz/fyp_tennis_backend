# Flutter Chat API Contract

This document describes the currently implemented backend contract for the chat and RAG features that a Flutter app can integrate against.

Relevant backend files:
- [`/home/hamza/personal/fyp_tennis_backend/src/api/chat.py`](/home/hamza/personal/fyp_tennis_backend/src/api/chat.py)
- [`/home/hamza/personal/fyp_tennis_backend/docs/chat_api_contract.md`](/home/hamza/personal/fyp_tennis_backend/docs/chat_api_contract.md)
- [`/home/hamza/personal/fyp_tennis_backend/src/main.py`](/home/hamza/personal/fyp_tennis_backend/src/main.py)

## Base Rules

All chat endpoints require:

- `Authorization: Bearer <token>`

Chat routes currently available:

- `POST /chat/start`
- `POST /chat/{session_id}/messages`
- `GET /chat/history`
- `GET /chat/history/{session_id}`
- `GET /chat/{stream_id}`

## Global Response Shape

Successful JSON responses usually follow:

```json
{
  "success": true,
  "message": "Some message",
  "data": {}
}
```

Error responses follow:

```json
{
  "success": false,
  "message": "Session expired"
}
```

Common error behavior from the backend:

- `401` -> `Session expired`
- `403` -> `Access denied`
- `500` -> `Internal server error`

## Chat API

### 1. Start New Chat

`POST /chat/start`

Starts a new chat session and returns the stream identifier for the first assistant response.

Content type:

- `multipart/form-data`

Fields:

- `message`: `string`, required
- `task_id`: `int`, optional
- `image`: file, optional, image only

Use `task_id` when the user starts a chat from a game details or game stats screen.

Example response:

```json
{
  "success": true,
  "message": "Chat session created",
  "data": {
    "sessionId": "6675ad43-2b81-4353-bdc1-018f2ecdb0f8",
    "streamId": "39236941-1015-4065-a488-e068c1473252",
    "streamUrl": "/chat/39236941-1015-4065-a488-e068c1473252"
  }
}
```

Flutter notes:

- Build this request with `MultipartRequest`.
- If the user selected an image, send it as the `image` part.
- If the user entered chat from a game page, include `task_id`.

### 2. Send Follow-Up Message In Existing Chat

`POST /chat/{session_id}/messages`

Adds a new user message to an existing session and returns a new stream for the assistant reply.

Content type:

- `multipart/form-data`

Fields:

- `message`: `string`, required
- `image`: file, optional

Example response:

```json
{
  "success": true,
  "message": "Chat message queued",
  "data": {
    "sessionId": "6675ad43-2b81-4353-bdc1-018f2ecdb0f8",
    "streamId": "a-new-stream-id",
    "streamUrl": "/chat/a-new-stream-id"
  }
}
```

App behavior:

- call this when the user sends the next message in an existing thread
- immediately connect to the returned `streamUrl`

### 3. Get Chat History List

`GET /chat/history`

Returns the authenticated user's chat session list.

Example response:

```json
{
  "success": true,
  "message": "Chat history fetched",
  "data": {
    "sessions": [
      {
        "id": "6675ad43-2b81-4353-bdc1-018f2ecdb0f8",
        "title": "How do I improve my serve?",
        "taskId": 42,
        "summary": "Recent tennis summary",
        "lastStreamId": "39236941-1015-4065-a488-e068c1473252",
        "lastMessagePreview": "Latest message preview",
        "updatedAt": "2026-04-27T00:00:00Z"
      }
    ],
    "contextSummary": "Readable user memory summary"
  }
}
```

Use this for:

- chat history screen
- recent conversation previews
- showing the user's accumulated memory summary

### 4. Get Full Chat Session

`GET /chat/history/{session_id}`

Returns the full transcript and session metadata for one chat session.

Example response:

```json
{
  "success": true,
  "message": "Chat session fetched",
  "data": {
    "session": {
      "id": "6675ad43-2b81-4353-bdc1-018f2ecdb0f8",
      "title": "How do I improve my serve?",
      "taskId": 42,
      "summary": "Recent tennis summary",
      "lastStreamId": "39236941-1015-4065-a488-e068c1473252"
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
            "fileSize": 12345
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
              "documentId": 1
            }
          ]
        },
        "attachments": [],
        "createdAt": "2026-04-27T00:00:10Z"
      }
    ]
  }
}
```

Use this for:

- reopening an existing chat
- rendering the full transcript
- optionally showing retrieval source metadata later

## SSE Streaming Contract

### 5. Stream Assistant Response

`GET /chat/{stream_id}`

Headers:

- `Authorization: Bearer <token>`
- `Accept: text/event-stream`

This endpoint streams assistant output using SSE.

Event types currently emitted:

- `started`
- `retrieval`
- `delta`
- `completed`
- `error`

Example stream:

```text
event: started
data: {"streamId":"39236941-1015-4065-a488-e068c1473252","sessionId":"6675ad43-2b81-4353-bdc1-018f2ecdb0f8"}

event: retrieval
data: {"sources":[{"type":"document","documentId":1}]}

event: delta
data: {"content":"Hello "}

event: delta
data: {"content":"from AceVision"}

event: completed
data: {"messageId":2}
```

Replay behavior:

- if a stream is already completed, calling the same `GET /chat/{stream_id}` returns:
  - `started`
  - one `delta` containing the full assistant message
  - `completed`

Flutter handling guidance:

- create an empty assistant bubble when `started` arrives
- append each `delta.content` to that bubble
- when `completed` arrives, mark the message as finished
- if `error` arrives, stop loading and show a failure state

## Access Rules

Current backend access behavior:

- `task_id` is optional on `POST /chat/start`
- if `task_id` is present, only the task owner, tagged opponent, or admin may access the session and its stream
- `GET /chat/history` only returns the authenticated user's own sessions

For Flutter:

- only send `task_id` when the chat should be scoped to a specific game
- expect `403` if the user tries to access a game-scoped session they do not own and are not tagged into

## Suggested Flutter Data Models

```dart
class ChatStartResponse {
  final bool success;
  final String message;
  final ChatStartData data;
}

class ChatStartData {
  final String sessionId;
  final String streamId;
  final String streamUrl;
}

class ChatHistoryResponse {
  final bool success;
  final String message;
  final ChatHistoryData data;
}

class ChatHistoryData {
  final List<ChatSessionSummary> sessions;
  final String? contextSummary;
}

class ChatSessionSummary {
  final String id;
  final String title;
  final int? taskId;
  final String? summary;
  final String? lastStreamId;
  final String? lastMessagePreview;
  final DateTime? updatedAt;
}

class ChatSessionDetailResponse {
  final bool success;
  final String message;
  final ChatSessionDetailData data;
}

class ChatSessionDetailData {
  final ChatSessionInfo session;
  final List<ChatMessageItem> messages;
}

class ChatSessionInfo {
  final String id;
  final String title;
  final int? taskId;
  final String? summary;
  final String? lastStreamId;
}

class ChatMessageItem {
  final int id;
  final String role;
  final String content;
  final Map<String, dynamic>? metadata;
  final List<ChatAttachmentItem> attachments;
  final DateTime? createdAt;
}

class ChatAttachmentItem {
  final int id;
  final String type;
  final String filename;
  final String mimeType;
  final int fileSize;
}

class ChatSseEvent {
  final String event;
  final Map<String, dynamic> data;
}
```

## Recommended Flutter Flow

### New chat

1. User enters message
2. Optional image selected
3. Optional `taskId` added if coming from game stats
4. Call `POST /chat/start`
5. Save `sessionId`
6. Open SSE on returned `streamUrl`
7. Render streamed assistant text from `delta`

### Existing chat

1. Load transcript with `GET /chat/history/{session_id}`
2. User sends new message
3. Call `POST /chat/{session_id}/messages`
4. Open returned SSE stream
5. Append streamed assistant response

### Chat list

1. Call `GET /chat/history`
2. Render `sessions`
3. Optionally show `contextSummary` in a memory or profile context area

## Multipart Field Names

For `POST /chat/start`:

- `message`
- `task_id`
- `image`

For `POST /chat/{session_id}/messages`:

- `message`
- `image`

## Current Limitations

These are true for the current backend implementation:

- admin prompt and document management is server-rendered HTML, not JSON API
- attachment metadata is returned, but there is no dedicated attachment download endpoint yet
- only image attachments are supported in chat
- no websocket support, only SSE
- no history pagination yet
- stream replay is basic for completed streams

## Useful Backend References

`src/api/chat.py` stream behavior:

```python
yield _sse("started", {"streamId": stream_id, "sessionId": chat_session.id})
yield _sse("retrieval", {"sources": retrieval["sources"]})
yield _sse("delta", {"content": piece})
yield _sse("completed", {"messageId": assistant_message.id})
```

`src/api/chat.py` session creation:

```python
return success_response(
    "Chat session created",
    {
        "sessionId": chat_session.id,
        "streamId": chat_stream.id,
        "streamUrl": f"/chat/{chat_stream.id}",
    },
)
```
