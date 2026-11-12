# OpenAI GPT-6 support through the Responses API (3.46.0)

Headline feature for 3.46.0 (feature cutoff ~2026-11-25). Approved 2026-11-11.
User report: [Discussion #928](https://github.com/goruck/home-generative-agent/discussions/928).

## Problem

Every tool turn on a GPT-6 model fails with HTTP 400: "Function tools with reasoning_effort are
not supported for gpt-6-luna in /v1/chat/completions. To use function tools, use /v1/responses or
set reasoning_effort to 'none'."

- `__init__.py` builds the OpenAI provider with `use_responses_api=False`, a guard added in
  `13fd42e9` (HA 2026.10 migration) so langchain-openai 1.6.7 would not move some model names to
  the Responses API. The integration was never tested on Responses.
- `_thinking_openai` (`core/utils.py`) sends only `minimal|low|medium|high`, so it can neither
  send `none` nor GPT-6's `xhigh`/`max`.
- The OpenAI model lists in `const.py` stop at `gpt-5`.

## What OpenAI documents (developers.openai.com, read 2026-11-11)

| Model | Effort values | Tools on Chat Completions |
|---|---|---|
| `gpt-6-astra` | low, medium, high, xhigh, max (`none` returns 400) | No: Responses only |
| `gpt-6.1-sol` | low, medium, high, xhigh, max (no `none`/`minimal`; default medium) | No: Responses only |
| `gpt-6.1-terra` | none, low, medium, high, xhigh, max | Only with `none` |
| `gpt-6-luna` | none, low, medium, high, xhigh, max (default medium) | Only with `none` |
| `gpt-5.4` and later (5.5, 5.6-*) | model-dependent; GPT-5.6 defaults to medium | Only with `none` ("Starting with GPT-5.4") |

- All current models take image input.
- Responses are stored at OpenAI by default; `store: false` turns that off. In stateless mode,
  reasoning items carry `encrypted_content` by default, and the docs say to replay every returned
  reasoning item, and to pass reasoning items back with tool outputs.
- `reasoning.mode` (`standard` | `pro`) exists on GPT-5.6 and GPT-6 models in Responses. Out of
  scope.

## Decisions

1. **Routing (approved).** An integration-owned predicate sends a model to the Responses API
   when it needs it: `gpt-6*`, `gpt-5.4` and later, langchain-openai's Responses-only list
   (`*-pro`, `gpt-5.6-sol`) and `*codex*`. Every other model stays on Chat Completions. It is set
   per call next to the model name, since one client serves every model. The OpenAI-compatible
   provider stays on Chat Completions.
2. **Reasoning effort.** Pass through `none|minimal|low|medium|high|xhigh|max` as the model
   accepts them. "Reasoning off" sends `none` only to models that accept it; for `gpt-6-astra`
   and `gpt-6.1-sol` it sends `low` (the lowest they accept). The form labels the choice
   "Off (GPT-6)"; `docs/configuration.md` says it is `low` on those two.
3. **Message format.** Before writing the feature, a spike confirms that Responses-shaped
   messages work with: text extraction (`_normalize_ai_content`, streaming deltas, schema-first),
   tool-call parsing, token counting, checkpoint replay on the next turn, and the fallback chain.
   `output_version` is the lever: `v0` keeps a reply's content to text blocks, so the history
   converts for Anthropic, Ollama and Gemini with no stripping step. (The agent graph also
   checkpoints each reply as text plus tool calls only, `agent/graph.py` `_call_model`.)
4. **Storage (approved).** `store=False`: nothing is kept at OpenAI. Same privacy as today on
   Chat Completions.
   **Reasoning replay (changed 2026-11-11 after the live test).** The plan was to replay the
   encrypted reasoning items from our checkpoint. Live, GPT-6 often returns several reasoning
   items in one reply (4-8 on a multi-step tool task), and the `v0` output format holds one per
   message: streaming concatenates their encrypted contents, and the next tool turn gets 400
   `invalid_encrypted_content` (reproduced on `gpt-6-luna`; OpenAI returns the encrypted copy
   with `store=False` whether or not `include` asks for it). `responses/v1` keeps the items
   apart and replayed cleanly over 15 live turns, but its blocks break the Ollama fallback.
   Decision (Lindo): don't replay. `core/openai_client.py` drops reasoning items from every
   Responses request and from the reply. Replay is a TODOS.md follow-up.
5. **Model lists.** Add the documented IDs to the OpenAI chat, image-analysis, summarization and
   Sentinel lists where they fit. The recommended default stays `gpt-5`.

## Paths to cover

Chat (streamed and schema-first), voice (streamed TTS, turn-start acknowledgement), the
PIN-held action flow, image analysis, summarization, Sentinel triage and discovery when they run
on OpenAI, the fallback chain, token counting.

## Tests

- Unit: routing predicate, effort mapping per model, normalization of Responses-shaped messages,
  a GPT-6 history converting for each fallback provider, `store=False` in the request.
- A recorded streaming fixture for a Responses tool turn.
- Live smoke test with Lindo's OpenAI key (never read from `core.config_entries`): a GPT-6 tool
  turn, a streamed voice reply, a PIN-held action, a mid-conversation fallback to Ollama, and an
  old Chat Completions thread continued on GPT-6.

Box field test (2026-11-12, HA 2026.11.0, Conversation on `gpt-6-luna` with an Anthropic
fallback): a tool turn, a streamed voice reply and a PIN-held action passed, with no 400 and no
fallback. A read-only multi-step task (read every temperature sensor and light, propose a
schedule) took 11-15 s at Extra-high effort and 44-49 s at Max, well inside the 120 s read
timeout and the 180 s turn limit, so the effort-timeout follow-up from the review was closed.

## Out of scope

`previous_response_id` (the checkpointer owns history), OpenAI built-in tools, `reasoning.mode`,
changing the recommended default model.

## Delivery

One PR, gstack review (model-provider plumbing), CHANGELOG `## [Unreleased]` entry, README model
list, then a pre-release offered on Discussion #928.
