# Flutter Delta Only

Only the new app-side changes are listed below. Everything else can stay as-is.

## 1. `GET /chat/history` session item

New field:

```json
{
  "lastAttachmentImageUrl": "/uploads/chat_attachments/serve.png"
}
```

What to do:
- add `lastAttachmentImageUrl` to the session summary model
- use it as the conversation preview image when present

## 2. `GET /chat/history/{session_id}` attachment object

New fields on each attachment:

```json
{
  "url": "/uploads/chat_attachments/serve.png",
  "viewUrl": "/uploads/chat_attachments/serve.png",
  "downloadUrl": "/uploads/chat_attachments/serve.png"
}
```

What to do:
- add these 3 fields to the attachment model
- use `url` or `viewUrl` to render the attached image in chat
- use `downloadUrl` if you expose save/share/open later

## 3. Assistant `metadata.sources` document citation shape

Old:

```json
{
  "type": "document",
  "documentId": 1
}
```

New:

```json
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
}
```

What to do:
- remove any client dependency on `documentId`
- update citation UI to show `title` and `pageRange`
- optionally show `governingBody`, `competition`, `seasonYear`
- open `viewUrl` when user taps a citation

## 4. User memory source entries

Example:

```json
{
  "type": "user_memory",
  "summary": "User has been working on serve placement and movement recovery.",
  "source": "chat_session"
}
```

What to do:
- accept this source type in the citation/source model
- no client-side dedup needed anymore; backend already returns unique memory entries

## 5. SSE `retrieval` event payload

Old:

```text
event: retrieval
data: {"sources":[{"type":"document","documentId":1}]}
```

New:

```text
event: retrieval
data: {"sources":[{"type":"document","title":"ITF Rules 2026","pageRange":"4-5","viewUrl":"/uploads/knowledge_documents/itf-rules-2026.pdf"}]}
```

What to do:
- update SSE retrieval parsing to use the new source shape
- do not expect `documentId`

## Minimal Dart model delta

```dart
class ChatSessionSummary {
  final String? lastAttachmentImageUrl;
}

class ChatAttachmentItem {
  final String url;
  final String viewUrl;
  final String downloadUrl;
}

class ChatSourceItem {
  final String type;
  final String? title;
  final String? governingBody;
  final String? competition;
  final int? seasonYear;
  final int? pageStart;
  final int? pageEnd;
  final String? pageRange;
  final int? lineStart;
  final int? lineEnd;
  final String? viewUrl;
  final String? downloadUrl;
  final String? summary;
  final String? source;
}
```

## UI delta checklist

- show `lastAttachmentImageUrl` in chat history when present
- render attachment images from `attachment.url`
- update source/citation widgets to use `title` + `pageRange`
- remove all `documentId` assumptions
