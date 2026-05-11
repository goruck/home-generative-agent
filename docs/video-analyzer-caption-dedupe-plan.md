# Video Analyzer Caption Deduplication Plan

## Problem

The video analyzer can send repeated notifications for the same low-value camera
activity. A recent `frontgate` example showed several notifications within a
short period for visually similar nighttime artifacts:

- no people visible in a black-and-white scene
- bright blur or light streaks across the walkway
- bright light near the driveway or top of frame
- quiet monochrome walkway scene

These are semantically close enough that the notification path should suppress
most repeats within a short window.

## Current Behavior

`VideoAnalyzer._is_anomaly()` searches prior `video_analysis` records and treats
the current caption as anomalous when any returned result has a score below
`VIDEO_ANALYZER_SIMILARITY_THRESHOLD`.

That means one weak match among the top search results can cause a notification,
even when the best match is highly similar. For notification deduplication, the
best matching recent caption is the important signal.

The image-level dHash gate is intentionally disabled. It adds image decode and
hashing work to the snapshot path, which is too expensive for Raspberry Pi 5
class hosts and can increase latency. This plan keeps dHash disabled.

The time-offset gate (`first_dt < now() - VIDEO_ANALYZER_TIME_OFFSET`) is
intentional: a snapshot whose filename timestamp falls outside the recency window
is always treated as novel regardless of caption similarity. This plan leaves
that gate unchanged.

## Score Semantics

`store.asearch` returns `r.score = 1 - cosine_distance` (already documented at
`agent/graph.py` line 690 and confirmed in the langgraph postgres store source).
Higher score means more similar: 1.0 is identical, 0.0 is orthogonal. `max` is
therefore the correct aggregator for "best match."

## Goals

- Reduce repeated low-value camera notifications without adding meaningful CPU
  cost.
- Keep the solution caption-first and compatible with constrained hosts.
- Preserve notifications for genuinely new subjects or actions.
- Make suppression decisions observable enough to tune with real logs.

## Non-Goals

- Do not re-enable dHash by default.
- Do not add heavyweight image comparison or local vision post-processing.
- Do not make broad changes to video analysis, face recognition, or snapshot
  capture.

## Proposed Changes

1. Fix the vector similarity decision.

   Use the best returned score instead of `any(score < threshold)`:

   ```python
   scores = [r.score for r in search_results if r.score is not None]
   best_score = max(scores, default=None)
   is_novel = best_score is None or best_score < threshold
   ```

   A high best score suppresses the notification even if lower-scoring results
   are also returned. Because `r.score` is a similarity value (higher = more
   similar), `max` correctly selects the closest stored caption.

   Rename `_is_anomaly` to `_is_caption_novel` in the same commit so the
   behavior matches the name.

2. Keep the search limit at 10.

   More candidates improve the best-score signal. Do not reduce the limit as
   part of this change.

   `_store_results` is called unconditionally in `_finalize()` — every analysis
   result is stored, not just those that triggered notifications. Filtering by
   "notification-worthy" records would require adding a notification-status field
   to the stored value and a query-time filter; treat that as a separate, explicitly
   scoped work item rather than part of this patch.

   Until that metadata exists, the first implementation still compares against
   all stored analyses for the camera. That is acceptable for the initial
   best-score bug fix, but it means suppression can be based on prior captions
   that did not themselves trigger notifications.

   Suppressed analyses must still be stored. The current `_finalize()` flow calls
   `_handle_notification()` and then `_store_results()`. Keep that behavior when
   `_handle_notification()` starts using a decision object so suppressed captions
   remain available for future deduplication.

   The vector similarity path does not apply a separate recency check against
   `VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC`. It relies on the existing
   `stale_snapshot` gate for truly old batches and on the natural recency
   distribution of search results. Only the lexical artifact fast path (step 6)
   enforces the dedupe window explicitly.

3. Return a structured novelty decision.

   Add a small decision object so notification behavior, debug logging, tests,
   and later metadata storage all use the same result:

   ```python
   @dataclass(frozen=True)
   class CaptionNoveltyDecision:
       notify: bool
       reason: Literal[
           "stale_snapshot", "no_match", "score_none",
           "score_below_threshold", "score_above_threshold",
           "artifact_bucket", "stale_match", "store_timeout",
       ]
       best_score: float | None = None
       matched_caption: str | None = None
       matched_age_seconds: int | None = None
   ```

   Using `Literal` for `reason` keeps the defined vocabulary in one place and
   lets pyright catch misspellings in both the implementation and tests.

   `matched_caption` is read from `r.value.get("content")` on the best-scoring
   search result. `matched_age_seconds` is derived from `r.created_at` compared
   against `dt_util.now()`.

   Decision reason semantics:

   - `stale_snapshot`: notify; first snapshot timestamp predates
     `VIDEO_ANALYZER_TIME_OFFSET` — the existing unchanged gate, fires before
     `store.asearch` is called
   - `no_match`: notify; vector search returned no candidate captions
   - `score_none`: notify; candidates existed but none had usable scores
   - `score_below_threshold`: notify; best usable score is below
     `VIDEO_ANALYZER_SIMILARITY_THRESHOLD`
   - `score_above_threshold`: suppress; best usable score is at or above
     `VIDEO_ANALYZER_SIMILARITY_THRESHOLD`
   - `artifact_bucket`: suppress; lexical artifact fast path matched and stored
     caption is within `VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC`
   - `stale_match`: notify; artifact bucket matched but stored caption is outside
     `VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC`
   - `store_timeout`: notify; `store.asearch` timed out

   Drop `malformed_score`. The list comprehension
   `[r.score for r in search_results if r.score is not None]` already excludes
   `None`, and the store type guarantees `r.score` is `float | None`, leaving no
   reachable path that produces a malformed score.

   `_handle_notification()` can use `decision.notify`, while logging and tests
   can assert the reason and match details. A later metadata patch can persist
   `notified`, `decision_reason`, and `matched_key` or `matched_caption` without
   reworking the helper API.

4. Add decision logging.

   Emit one debug log line per notification decision with:

   - camera id/name
   - current caption
   - best score
   - matched prior caption, if available
   - match age, if available
   - threshold
   - decision: notify or suppress
   - reason: no match, score below threshold, score above threshold, artifact
     bucket, or stale match

   Add this before normalization and artifact-bucket suppression so real log data
   can inform tuning of those later steps.

   After artifact-bucket suppression lands, include the final lexical fields in
   the same debug line: artifact bucket, subject-present flag, negated-human flag,
   and whether known face recognition forced notification.

5. Add cheap caption normalization (lexical path only).

   Normalize captions using inexpensive string operations for the lexical
   fast-path comparison in step 6. Do not apply normalization to text passed to
   the vector store or to the query in `store.asearch` — the embedding model
   works on natural language and normalization can degrade semantic distance
   calculations.

   Normalization operations:

   - lowercase
   - strip camera/notification boilerplate if present
   - collapse whitespace and punctuation noise
   - normalize common artifact terms such as `bright light`, `glare`, `streak`,
     `blur`, `black and white`, `monochrome`, and `night scene`

6. Add a lexical fast path for repeated low-value artifacts.

   If the current and prior captions fall into the same low-value artifact bucket
   and neither mentions a human, package, animal, or vehicle subject, suppress the
   notification within `VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC`.

   Add a dedicated caption dedupe window constant, initially:

   ```python
   VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC = 1800
   ```

   Keep this separate from `VIDEO_ANALYZER_TIME_OFFSET`. The time offset handles
   stale snapshot batches; the caption dedupe window handles repeated
   notification suppression.

   Example artifact bucket terms:

   - bright light
   - light streak
   - glare
   - blur
   - monochrome or black-and-white scene
   - no people visible

   Example subject terms that should prevent cheap suppression:

   - person, people, man, woman, child
   - package, box, delivery
   - car, truck, vehicle
   - known face names from face recognition

   The subject list is the critical safety valve — false negatives (suppressing
   real events) are the greater risk. Keep the list conservative and easy to
   extend from config or const.

   Handle negated subject phrases explicitly. Phrases such as `no people visible`,
   `no person visible`, `no one visible`, and `nobody visible` should count as
   human absence, not human presence. This matters because `No people are visible
   in the black and white scene` is one of the repeated artifact cases the fast
   path should be able to suppress.

   The renamed novelty helper should accept `recognized_names: list[str]` directly
   rather than reaching into `self._last_recognized`. The async novelty method will
   still call `store.asearch()` and inspect timestamps; keep only the artifact and
   subject text classification helpers pure so they are straightforward to unit
   test.

   The call site in `_handle_notification` is the only place instance state needs
   to be accessed:

   ```python
   decision = await self._is_caption_novel(
       camera_name, msg, first_snapshot,
       recognized_names=list(self._last_recognized.get(camera_id, [])),
   )
   ```

7. Add focused regression tests.

   The existing `test_video_analyzer_priority.py` covers queue management,
   semaphore behavior, and VLM configuration. Novelty/suppression tests should
   be added as a new section in that file or in a new
   `test_video_analyzer_novelty.py`.

   Cover screenshot-like captions:

   - `A bright horizontal blur streaks across the walkway.`
   - `A bright light appears near the top left of the driveway at night.`
   - `A bright light streaks across the walkway at night.`
   - `No people are visible in the black and white scene.`
   - `A white vehicle is parked beyond the fences in the monochrome scene.`

   Expected behavior:

   - a high best vector score suppresses notification
   - no prior match notifies
   - low best score notifies
   - one high score plus several low scores suppresses notification
   - repeated artifact-bucket captions suppress within the recency window
   - captions with real subject/action changes still notify
   - a caption containing both artifact terms and a vehicle subject is not
     suppressed by the lexical fast path
   - `No people are visible...` followed by `A person walks up the path...`
     notifies even when both captions mention the same walkway or night scene
   - negated human phrases such as `no people visible` do not block artifact
     suppression
   - a stale or backlogged batch whose first snapshot is older than
     `VIDEO_ANALYZER_TIME_OFFSET` still notifies, preserving the current
     time-offset gate behavior
   - no search results, all `score is None`, and malformed score data notify with
     distinct decision reasons
   - `store.asearch` timeout treats the event as novel and notifies

## Implementation Order

1. Fix best-score threshold handling, rename `_is_anomaly` to
   `_is_caption_novel`, and add unit tests for the current bug. (One commit.)
2. Add `CaptionNoveltyDecision` dataclass.
3. Add decision logging (driven by the `CaptionNoveltyDecision` result).
4. Add lexical normalization and artifact-bucket suppression, including negated
   subject handling and known-face bypass.
5. Add recent-notification metadata/filtering as a separate, explicitly scoped
   work item if the store path supports it cleanly.
6. Tune threshold and artifact terms from real debug logs.

## Validation

Run focused tests after the first patch:

```bash
./hga/bin/pytest tests/custom_components/home_generative_agent/test_video_analyzer_priority.py
```

If helper functions are added outside the existing test coverage, add or extend a
dedicated video analyzer regression test file and run the focused suite plus
Ruff on touched files.
