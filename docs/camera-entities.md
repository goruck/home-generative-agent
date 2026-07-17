# Camera Entities, Dashboards, and Face Recognition

The integration creates image and sensor entities for each configured camera, and provides services for on-demand analysis and proactive video monitoring. Face recognition can optionally identify people in camera frames.

- [Entity Overview](#entity-overview)
- [On-Demand Snapshot Analysis](#on-demand-snapshot-analysis)
- [Proactive Video Analysis](#proactive-video-analysis)
  - [How triggers work by camera type](#how-triggers-work-by-camera-type)
  - [Motion → camera resolution](#motion--camera-resolution)
  - [Notification modes](#notification-modes)
  - [Caption deduplication](#caption-deduplication)
  - [VLM quality requirement](#vlm-quality-requirement)
  - [Resource management](#resource-management)
  - [Advanced options](#advanced-options)
- [Lovelace Dashboards](#lovelace-dashboards)
- [Automations](#automations)
- [Events and Signals](#events-and-signals)
- [Face Recognition](#face-recognition)

---

## Entity Overview

For each configured camera the integration registers:

| Entity | ID pattern | Description |
|---|---|---|
| Image | `image.<camera_slug>_last_event` | Most recent snapshot published by the analyzer |
| Sensor | `sensor.<camera_slug>_recognized_people` | People recognized in the last frame |

**Sensor attributes:**

| Attribute | Description |
|---|---|
| `recognized_people` | Names from face recognition |
| `summary` | AI description of the last frame |
| `latest_path` | Filesystem path of the published image |
| `count` | Number of recognized people |
| `last_event` | Timestamp of the last event |
| `camera_id` | Source camera entity ID |

**Diagnostic sensor:**

`sensor.tool_index_status` (entity category: diagnostic) shows the state of the RAG tool index:

| State | Meaning |
|---|---|
| `indexing` | First-run embedding in progress |
| `ready` | Index built; tools retrieved per-turn by semantic search |
| `failed` | Embedding provider unreachable; agent falls back to all tools |
| `unknown` | Index state not yet reported |

> HA needs write access to your snapshots location (default: `/media/snapshots`) and your camera entities must exist in HA (`camera.*`).

---

## On-Demand Snapshot Analysis

The `home_generative_agent.save_and_analyze_snapshot` service captures a fresh snapshot, analyzes it, and publishes it as the latest event.

**Fields:**

| Field | Required | Default | Description |
|---|---|---|---|
| `entity_id` | yes | — | One or more `camera.*` entities |
| `protect_minutes` | no | `30` | Protect the new file from pruning |

**Example (Developer Tools → Actions):**

```yaml
service: home_generative_agent.save_and_analyze_snapshot
target:
  entity_id:
    - camera.frontgate
    - camera.backyard
data:
  protect_minutes: 30
```

**Example button card:**

```yaml
type: button
name: Refresh Frontgate
icon: mdi:camera
tap_action:
  action: call-service
  service: home_generative_agent.save_and_analyze_snapshot
  target:
    entity_id: camera.frontgate
```

---

## Proactive Video Analysis

When enabled, the integration watches for motion and recording events and automatically triggers snapshot analysis on the associated camera. Results are stored in the database for use by the agent, and optional notifications are sent to the mobile app.

### How triggers work by camera type

Different camera integrations surface motion in different ways. The video analyzer handles all of them:

#### Axis cameras (VMD-based)

The Axis HA integration creates `binary_sensor.<name>_vmd<N>` entities that pulse briefly when motion is detected. The analyzer listens for these rising edges, starts a snapshot loop at 3-second intervals, and cancels it when the sensor returns to `off`, then flushes the collected frames through the VLM pipeline as a batch.

The `_vmd<N>` suffix is stripped automatically during resolution (e.g. `binary_sensor.frontgate_vmd1` → `camera.frontgate`). The camera entity itself does not need to enter any particular state.

#### Ring cameras via ring-mqtt

The [ring-mqtt add-on](https://github.com/tsightler/ring-mqtt) bridges Ring devices to HA over MQTT. It creates:

- `binary_sensor.<name>_motion` — the doorbell's or camera's own motion sensor. This stays `on` for a configurable duration (10–180 s, default 180 s) then reverts to `off` automatically.
- `camera.<name>_snapshot` — the associated camera entity for snapshots.
- `select.<name>_event_select` — an event-history selector whose `eventId` attribute changes on every Ring event.

The analyzer starts a snapshot loop when `binary_sensor.<name>_motion` goes `on` and cancels it when it goes `off`, collecting frames every 3 seconds across the full motion window before flushing as a batch.

**Battery-powered Ring cameras** (e.g. Ring Battery Doorbell Plus) may not push `binary_sensor.*_motion` promptly or reliably over MQTT. For these, the analyzer also listens for `eventId` changes on `select.<name>_event_select` entities. When the `eventId` changes, the associated camera is resolved (same three-tier lookup as motion sensors) and the same snapshot loop starts. Because ring-mqtt sends no corresponding "event over" signal on this entity, the loop runs for a fixed 30-second window and then flushes the collected frames as a batch; another `eventId` change during the window extends it (total window length is capped at 5 minutes so continuous events cannot defer the flush forever).

Both triggers coexist safely via loop ownership: the window timer only ever ends a loop that `event_select` itself started. If the motion sensor started the loop (or fires `on` while an `event_select` loop is running), the motion sensor owns the lifecycle — the window timer is retired and the loop runs until motion returns to `off`, exactly as before. Retained-state replays are ignored: an `eventId` that appears when the entity leaves `unknown`/`unavailable` (HA startup, MQTT broker reconnect, ring-mqtt add-on restart) carries the *previous* event and does not trigger. No configuration is needed; the `event_select` path is detected automatically.

**Duplicate-frame collapse on battery cameras.** ring-mqtt's `Auto` snapshot mode is power-aware: wired cameras refresh their snapshot image every 30 s, but battery cameras only every 600 s. A 30-second capture window therefore cannot yield distinct frames on a battery camera — it would batch ~10 identical copies of one frame, each costing a VLM call. To prevent this, the perceptual-hash uniqueness gate is always active for frames captured by a loop the `event_select` path started, regardless of the global uniqueness option (the first frame of each window is always accepted, so every capture window still produces at least one analysis). The forced gate persists even when a lagging motion sensor takes over the loop mid-window — the underlying frames are still 600-second interval snapshots — and is retired when the loop stops. Frames the gate skips are still registered for normal snapshot retention, so duplicates never accumulate on disk. Loops started by a motion sensor keep the opt-in behavior.

**Stale-snapshot guard.** The ring-mqtt interval snapshot can silently stop updating on battery cameras, after which MQTT serves the last retained frame indefinitely (observed in the field at 5 days old). Before every capture, the analyzer checks the snapshot camera's `timestamp` attribute (epoch seconds of the last published frame); if the frame is older than 30 minutes, the capture is skipped and counted as a snapshot failure (escalating from WARNING to ERROR on repeats, as for issue #464) instead of sending days-old imagery to the VLM. The failure is recorded once per staleness episode — not once per 3-second loop iteration — where a changed frozen timestamp counts as a new episode and an ongoing freeze is re-recorded hourly (so a multi-day freeze stays visible in the hourly metrics and the escalation can reach ERROR); the episode clears when a fresh timestamp appears. Restarting the ring-mqtt add-on refreshes the snapshot. The guard only applies to cameras with a ring-mqtt `event_select` sibling (matched by naming, the motion→camera override map, or a `select.*_event_select` entity on the same HA device — the same device-registry resolution the trigger path uses for renamed entities): other integrations may publish a `timestamp` attribute with different semantics, and those cameras — as well as any camera whose timestamp is not a plausible current epoch (non-numeric, millisecond-scale, far-future) — are never marked stale.

**Important:** ring-mqtt cameras are resolved using the [device registry](#motion--camera-resolution) rather than name matching. This is necessary because Ring Alarm motion sensors (indoor PIRs) often share a name prefix with the doorbell — e.g. `binary_sensor.front_door_motion` (a wall PIR) and `binary_sensor.front_door_motion_3` (the doorbell). The device registry approach selects the sensor that shares a device entry with the camera, avoiding false bindings to unrelated sensors.

Ring cameras do not expose a `recording` state in HA via ring-mqtt, so the recording-state path described below does not apply.

#### Cameras with recording state (UniFi Protect, generic)

Some integrations (e.g. UniFi Protect) expose cameras that enter a `recording` HA state when an event is detected, without necessarily firing a corresponding `binary_sensor.*` motion event. The analyzer handles this via an event-driven recording loop:

- When a camera enters `recording` state, a snapshot loop starts at 3-second intervals.
- When the camera exits `recording` state, the loop is cancelled and collected frames are flushed as a batch.
- If a `binary_sensor.*` motion event fires for the same camera while a recording loop is already running, the motion event takes ownership — the motion loop's `off` transition then controls the queue flush.

A periodic poll (every 1.5 s) also runs as a safety net for cameras that were already in `recording` state when the integration started (no `state_changed` event fires in that case). It skips any camera already covered by a motion or recording loop.

#### Official Ring HA integration

The official Ring HA integration communicates via the Ring cloud. Camera entities update their `last_video_id` and `video_url` attributes on a 1-minute poll cycle once a recording is ready — they do not enter a `recording` state. Motion binary sensors fire immediately via a Ring WebSocket push and revert after `expires_in` seconds (controlled by Ring's API). The ring-mqtt add-on is generally preferred over the official integration for proactive analysis because it provides more reliable and immediate motion signals.

---

### Motion → camera resolution

When a `binary_sensor.*` motion event (or a `select.*_event_select` `eventId` change) fires, the integration resolves the associated camera using a three-tier lookup:

**Tier 1 — Explicit override map** (checked first)

Configure overrides in **Global Options → Motion sensor → camera overrides**. One pair per line:

```
binary_sensor.front_door_motion_3: camera.front_door_snapshot
binary_sensor.backyard_pir: camera.backyard
```

Use this when automatic resolution picks the wrong camera, or when you want to pin a sensor to a specific camera regardless of other logic. Leave blank to rely on tiers 2 and 3.

The hardcoded `VIDEO_ANALYZER_MOTION_CAMERA_MAP` constant in `const.py` is the code-level fallback if nothing is configured in Global Options.

**Tier 2 — Device registry** (recommended path for most modern integrations)

Finds the `camera.*` entity registered to the same HA device as the motion sensor. This is how Axis, ring-mqtt, and most well-integrated camera systems are resolved. It correctly handles naming collisions — for example, a Ring Alarm wall PIR and a Ring doorbell that share a name prefix are on different HA devices, so only the doorbell's own sensor binds to the doorbell camera.

Returns no result if the sensor has no device link, the device has no camera entities, or the device has more than one camera (ambiguous).

**Tier 3 — Name heuristics** (fallback for integrations without device links)

Applied in order:

1. Direct substitution: `binary_sensor.X` → `camera.X`
2. VMD-suffix strip: `binary_sensor.X_vmd1` → `camera.X`
3. `_motion`-suffix strip: `binary_sensor.X_motion` → `camera.X` or `camera.X_snapshot`

For `select.X_event_select` entities the heuristic strips the `_event_select` suffix instead: `select.X_event_select` → `camera.X` or `camera.X_snapshot`.

If none of the three tiers resolves to an existing camera entity, the motion event is silently ignored for that sensor.

---

### Notification modes

Set in **Global Options → Video analyzer mode**:

| Mode | Behavior |
|---|---|
| `disable` | Video analysis off |
| `notify_on_anomaly` | Notify only for semantically novel scenes (uses caption deduplication) |
| `always_notify` | Notify on every analyzed event |

---

### Caption deduplication

Active only in `notify_on_anomaly` mode. Repeated low-value notifications are suppressed using two complementary mechanisms:

**Semantic dedup:** The new caption is compared against recent captions using vector similarity. If the score meets or exceeds the similarity threshold (default 0.85), the notification is withheld. A 30-minute window is used; notifications for scenes with real subjects (people, vehicles, packages, animals) are always preserved. If a subject was last seen more than 30 minutes ago, notification resumes even if the similarity score is high.

**Lexical fast path:** A 30-minute window suppresses repeated artifact captions (nighttime glare, monochrome blur, empty walkway descriptions) even when the vector score falls below the threshold.

---

### VLM quality requirement

Semantic deduplication in `notify_on_anomaly` mode depends on the VLM producing **consistent captions** for the same scene. If two snapshots of an unchanged porch are described differently, the similarity score will be low and every frame will appear novel — resulting in a notification per snapshot.

Small models like `moondream` are too inconsistent for reliable dedup. Use a model with tested caption stability: `gemma3:4b` (the smallest tested option — reproduces the same core scene across repeated captions of identical frames) or a **7B-class vision model** such as `qwen2.5vl:7b`. The default Ollama VLM model is `qwen3-vl:8b`.

If you are using `always_notify` mode, VLM consistency affects caption quality but not notification volume.

---

### Resource management

The video pipeline enforces a per-entry semaphore that limits concurrent VLM and summary model calls (default: 1, sequential). Frames that cannot acquire the semaphore within 30 s are dropped. If a chat turn starts while the video pipeline is waiting for the model, it waits briefly for the chat turn to complete before dropping the frame — avoiding GPU contention. The video token budget is intentionally capped (256 tokens for VLM scene descriptions, 128 tokens for summaries).

---

### Advanced options

**In Global Options** (gear icon on the integration page):

| Option | Default | Description |
|---|---|---|
| `video_analyzer_mode` | `disable` | Notification mode: disable / notify_on_anomaly / always_notify |
| `video_analyzer_uniqueness_enabled` | `false` | Enable perceptual hash (dHash) pre-filter to skip visually identical frames before VLM analysis. **Caveat:** drops near-duplicate snapshots, removing the visual continuity the summary model uses to narrate motion. Only enable if a nearly-static scene generates excessive duplicates and you accept that motion context may be lost. |
| `video_analyzer_motion_camera_map` | *(empty)* | Explicit `binary_sensor: camera` overrides, one per line. Use when automatic resolution picks the wrong camera. |

**In the Camera Image Analysis feature subentry:**

| Option | Default | Description |
|---|---|---|
| `video_model_semaphore` | `1` | Concurrent video model call limit. Increase only if your server has dedicated GPU headroom. |
| `model_provider_uncontended` | `false` | (Global Options) Bypass all local gates when the model server has dedicated capacity. |

![Video analysis notification example](../assets/video-analysis-screenshot.jpeg)

---

## Lovelace Dashboards

Replace `frontgate` with your camera's slug in all examples below.

**Simple image + summary (two cards per camera):**

```yaml
type: grid
columns: 2
square: false
cards:
  - type: vertical-stack
    cards:
      - type: picture-entity
        entity: image.frontgate_last_event
        show_name: false
        show_state: false
      - type: markdown
        content: |
          **Summary**
          {{ state_attr('image.frontgate_last_event', 'summary') or '—' }}

          **Recognized**
          {% set names = state_attr('image.frontgate_last_event', 'recognized_people') or [] %}
          {{ names | join(', ') if names else 'None' }}
```

**All-cameras grid view:**

```yaml
title: Cameras
path: cameras
cards:
  - type: grid
    columns: 2
    square: false
    cards:
      # Repeat this block for each camera slug
      - type: vertical-stack
        cards:
          - type: picture-entity
            entity: image.frontgate_last_event
            show_name: false
            show_state: false
          - type: markdown
            content: |
              **Summary**
              {{ state_attr('image.frontgate_last_event', 'summary') or '—' }}

              **Recognized**
              {% set names = state_attr('image.frontgate_last_event', 'recognized_people') or [] %}
              {{ names | join(', ') if names else 'None' }}
```

**Overlay text on image:**

```yaml
type: picture-elements
image: /api/image_proxy/image.frontgate_last_event
elements:
  - type: markdown
    content: >
      {% set names = state_attr('image.frontgate_last_event', 'recognized_people') or [] %}
      **{{ names | join(', ') if names else 'None' }}**
    style:
      top: 6%
      left: 50%
      width: 92%
      color: white
      text-shadow: 0 0 6px rgba(0,0,0,0.9)
      transform: translateX(-50%)
  - type: state-label
    entity: image.frontgate_last_event
    attribute: summary
    style:
      bottom: 6%
      left: 50%
      width: 92%
      color: white
      text-shadow: 0 0 6px rgba(0,0,0,0.9)
      transform: translateX(-50%)
```

---

## Automations

**Notify when people are recognized on any camera:**

```yaml
alias: Camera recognized people
mode: parallel
trigger:
  - platform: state
    entity_id:
      - sensor.frontgate_recognized_people
      - sensor.playroomdoor_recognized_people
      - sensor.backyard_recognized_people
condition: []
action:
  - variables:
      ent: "{{ trigger.entity_id }}"
      cam: "{{ state_attr(ent, 'camera_id') }}"
      names: "{{ state_attr(ent, 'recognized_people') or [] }}"
      summary: "{{ state_attr(ent, 'summary') or 'An event occurred.' }}"
      image_entity: "image.{{ cam.split('.')[-1] }}_last_event"
  - service: notify.mobile_app_phone
    data:
      title: "Camera: {{ cam }}"
      message: >
        {{ summary }}
        {% if names %} Recognized: {{ names | join(', ') }}.{% endif %}
      data:
        image: >
          {{ state_attr(image_entity, 'entity_picture') }}
```

---

## Events and Signals

**`hga_last_event_frame`** is fired whenever a new latest frame is published:

```json
{
  "camera_id": "camera.frontgate",
  "summary": "A person approaches the gate...",
  "path": "/media/snapshots/camera_frontgate/snapshot_YYYYMMDD_HHMMSS.jpg",
  "latest": "/media/snapshots/camera_frontgate/_latest/latest.jpg"
}
```

Internal dispatcher signals (most users will not need these directly — platform entities update automatically):

| Signal | Updates |
|---|---|
| `SIGNAL_HGA_NEW_LATEST` | `image.*_last_event` |
| `SIGNAL_HGA_RECOGNIZED` | `sensor.*_recognized_people` |

---

## Face Recognition

Face recognition requires the optional [face-service](https://github.com/goruck/face-service) external service. See [Installation — Optional Apps](installation.md#optional-apps) for setup.

### Enroll via Service

**Service:** `home_generative_agent.enroll_person`

```yaml
service: home_generative_agent.enroll_person
data:
  name: "Eva"
  file_path: "/media/faces/eva_face.jpg"
```

The file must be inside Home Assistant's `/media` folder so it is accessible to the integration.

### Enroll via Dashboard Card

After registering the `hga-enroll-card.js` resource (see [Installation](installation.md#optional-apps)), add the card to any dashboard:

```yaml
type: custom:hga-enroll-card
title: Enroll Person
endpoint: /api/home_generative_agent/enroll
```

Use the file picker or drag-and-drop to upload one or more images. The card enrolls images that contain a detectable face and skips those that do not.
