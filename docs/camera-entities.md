# Camera Entities, Dashboards, and Face Recognition

The integration creates image and sensor entities for each configured camera, and provides services for on-demand analysis and proactive video monitoring. Face recognition can optionally identify people in camera frames.

- [Entity Overview](#entity-overview)
- [On-Demand Snapshot Analysis](#on-demand-snapshot-analysis)
- [Proactive Video Analysis](#proactive-video-analysis)
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

Enable proactive video scene analysis from cameras visible to Home Assistant. When enabled, motion detection triggers analysis, results are stored in the database for use by the agent, and optional notifications are sent to the mobile app. Anomaly detection can be enabled to send notifications only for semantically novel scenes.

**Caption deduplication:** In anomaly mode, repeated low-value notifications are suppressed. If the new caption is semantically close to a recent caption (vector similarity ≥ 0.89), the notification is withheld. A 30-minute lexical fast path additionally suppresses repeated artifact captions (nighttime glare, monochrome blur scenes, empty walkway descriptions) even when the vector score falls below the threshold. Notifications for scenes with real subjects (people, vehicles, packages, animals) are always preserved — and if a subject was last seen more than 30 minutes ago, notification resumes even if the vector score is high.

**Resource management on edge deployments:** The video pipeline enforces a per-entry semaphore that limits concurrent VLM and summary model calls (default: 1, sequential). Frames that cannot acquire the semaphore within 30 s are dropped. If a chat turn starts while the video pipeline is waiting for the model, it waits briefly for the chat turn to complete before dropping the frame — avoiding GPU contention. The video token budget is intentionally capped (256 tokens for VLM scene descriptions, 128 tokens for summaries).

**Advanced options** (in the Camera Image Analysis feature subentry):

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
