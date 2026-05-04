# Rally-First Stats API Contract

This document defines the new stats contract the app should use.

## Summary

- The canonical match-analysis response is now rally-first.
- Every public stat is attached to a rally and to `p1` or `p2`.
- The API no longer returns public unassigned analytics buckets.
- Every public attribution includes `attribution_confidence`.
- Each rally exposes both frame ranges and second ranges so the app can seek video directly.

## Player mapping

- `p1`: opponent / top-court player
- `p2`: owner / bottom-court player

These labels are stable across the payload.

## Main endpoint

`GET /all-stats/{task_id}`

Returns:

```json
{
  "success": true,
  "data": {
    "schema_version": 1,
    "task": {
      "id": 12,
      "name": "match-1",
      "status": "completed",
      "description": "Video processed successfully"
    },
    "video": {
      "source_video_path": "/uploads/match-1.mp4",
      "processed_video_path": "/output/output_12_123.mp4",
      "minimap_path": "/output/output_12_minimap_123.mp4",
      "thumbnail_path": "/output/output_12_thumbnail_123.jpg",
      "fps": 29.97,
      "total_frames": 4820
    },
    "players": {
      "p1": {
        "role": "opponent",
        "display": {
          "kind": "user",
          "userId": "uuid",
          "atTag": "@player1",
          "name": "Player One"
        }
      },
      "p2": {
        "role": "owner",
        "display": {
          "kind": "user",
          "userId": "uuid",
          "atTag": "@player2",
          "name": "Player Two"
        }
      }
    },
    "summary": {
      "total_rallies": 8,
      "total_shots": 54
    },
    "rallies": [
      {
        "rally_id": "rally_0",
        "scene_index": 0,
        "start_frame": 120,
        "end_frame": 260,
        "start_time_sec": 4.004,
        "end_time_sec": 8.675,
        "duration_sec": 4.671,
        "video": {
          "source_video_path": "/uploads/match-1.mp4",
          "processed_video_path": "/output/output_12_123.mp4",
          "minimap_path": "/output/output_12_minimap_123.mp4",
          "thumbnail_path": "/output/output_12_thumbnail_123.jpg",
          "playback_start_frame": 120,
          "playback_end_frame": 260,
          "playback_start_time_sec": 4.004,
          "playback_end_time_sec": 8.675
        },
        "summary": {
          "shot_count": 6,
          "bounce_count": 6,
          "serve_count": 1,
          "direction_change_count": 5
        },
        "shared": {
          "ball_track": [
            {
              "frame": 120,
              "time_sec": 4.004,
              "position": [810.1, 1420.4]
            }
          ],
          "ball_bounces": [
            {
              "frame": 132,
              "time_sec": 4.404,
              "position": [845.0, 1602.0],
              "is_serve": true,
              "serve_type": "corner",
              "player": "p1",
              "attribution_confidence": 0.95,
              "attribution_method": "legacy-detected"
            }
          ],
          "direction_changes": [
            {
              "frame": 125,
              "time_sec": 4.171,
              "position": [804.2, 1501.0]
            }
          ]
        },
        "players": {
          "p1": {
            "role": "opponent",
            "display": {
              "kind": "user",
              "userId": "uuid",
              "atTag": "@player1",
              "name": "Player One"
            },
            "heatmap": {
              "image_path": "/output/output_12_rally_0_p1_heatmap.png",
              "point_count": 48
            },
            "positions": [
              {
                "frame": 120,
                "time_sec": 4.004,
                "bbox": [320.0, 90.0, 384.0, 244.0],
                "court_position": [352.4, 244.0]
              }
            ],
            "speed_stats": [
              {
                "bounce_frame": 188,
                "time_sec": 6.273,
                "speed_kmh": 91.2,
                "time_diff_sec": 0.42,
                "distance_m": 10.64,
                "shot_type": "forehand",
                "player": "p1",
                "attribution_confidence": 0.7,
                "attribution_method": "origin-side"
              }
            ],
            "serve_stats": {
              "serves": [
                {
                  "bounce_frame": 132,
                  "time_sec": 4.404,
                  "bounce_position": [845.0, 1602.0],
                  "origin_frame": 125,
                  "origin_position": [804.2, 1501.0],
                  "ball_track": [
                    {
                      "frame": 125,
                      "time_sec": 4.171,
                      "position": [804.2, 1501.0]
                    }
                  ],
                  "serve_type": "corner",
                  "player": "p1",
                  "attribution_confidence": 0.95,
                  "attribution_method": "legacy-detected"
                }
              ],
              "summary": {
                "total": 1,
                "t": 0,
                "body": 0,
                "corner": 1,
                "wide": 0,
                "bucket": 0,
                "fault": 0
              }
            },
            "court_analysis": {
              "ball_bounces": [],
              "ball_points": [],
              "ball_track": [],
              "forehand_backhand": {
                "forehand": 1,
                "backhand": 0,
                "unknown": 0
              }
            }
          },
          "p2": {
            "role": "owner",
            "display": {
              "kind": "user",
              "userId": "uuid",
              "atTag": "@player2",
              "name": "Player Two"
            },
            "heatmap": {
              "image_path": "/output/output_12_rally_0_p2_heatmap.png",
              "point_count": 52
            },
            "positions": [],
            "speed_stats": [],
            "serve_stats": {
              "serves": [],
              "summary": {
                "total": 0,
                "t": 0,
                "body": 0,
                "corner": 0,
                "wide": 0,
                "bucket": 0,
                "fault": 0
              }
            },
            "court_analysis": {
              "ball_bounces": [],
              "ball_points": [],
              "ball_track": [],
              "forehand_backhand": {
                "forehand": 0,
                "backhand": 0,
                "unknown": 0
              }
            }
          }
        }
      }
    ]
  }
}
```

## App usage

### Rally list

Use `data.rallies` as the main list for the match-analysis screen.

### Video playback

When opening a rally:

- use `rally.video.processed_video_path` for annotated playback when desired
- use `rally.video.source_video_path` for the original match video when desired
- seek to `rally.video.playback_start_time_sec`
- stop or clip at `rally.video.playback_end_time_sec`

### Heatmaps

Each rally has one saved heatmap per player:

- `rally.players.p1.heatmap.image_path`
- `rally.players.p2.heatmap.image_path`

### Serve stats

Serve events are available at:

- `rally.players.p1.serve_stats.serves`
- `rally.players.p2.serve_stats.serves`

Each serve includes:

- `bounce_position`
- `origin_position`
- `ball_track`
- `serve_type`
- `attribution_confidence`

## Serve taxonomy

The backend now returns these serve types:

- `t`
- `body`
- `corner`
- `wide`
- `bucket`
- `fault`

`fault` means the bounce point is outside the service box.

## Projection endpoints

These endpoints now return rally-aware projections of the same canonical payload:

- `GET /get_video_paths/{task_id}`
- `GET /get_speed_stats/{task_id}`
- `GET /get_ball_track/{task_id}`
- `GET /get_bounces/{task_id}`
- `GET /get_direction_change_indices/{task_id}`
- `GET /get_player_positions/{task_id}`
- `GET /thumbnail/{task_id}`
- `GET /rally_stats/{task_id}`
- `GET /serve_stats/{task_id}`
- `GET /player_heatmaps/{task_id}`

## Legacy task conversion

Admin-only backfill endpoint:

`POST /admin/tasks/{task_id}/backfill-rally-analysis`

Behavior:

- protected; admin bearer auth is required
- does not rerun video processing
- reads legacy stored rows and writes the new rally-first analysis row
- safe to retry for the same task

Response:

```json
{
  "success": true,
  "data": {
    "task_id": 12,
    "schema_version": 1,
    "summary": {
      "total_rallies": 8,
      "total_shots": 54
    }
  }
}
```

## Migration note for the app

Old assumptions that are no longer valid:

- stats are no longer primarily match-wide blobs
- public `unknown` / `unassigned` player buckets should not be expected
- serve types are no longer limited to broad `t/body/wide` buckets
- video playback should use rally start/end timing from each rally entry, not ad hoc frame slicing on the client
