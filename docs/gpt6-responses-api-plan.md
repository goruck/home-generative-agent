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
   and `gpt-6.1-sol` it sends `low` (the lowest they accept) and the options form says so.
3. **Message format.** Before writing the feature, a spike confirms that Responses-shaped
   messages work with: text extraction (`_normalize_ai_content`, streaming deltas, schema-first),
   tool-call parsing, token counting, checkpoint replay on the next turn (reasoning items kept for
   OpenAI), and the fallback chain (reasoning items stripped before the history goes to
   Anthropic, Ollama or Gemini). `output_version` is the expected lever.
4. **Storage (approved).** `store=False`. Reasoning still carries across turns from the
   encrypted reasoning items in our own checkpoint, and nothing is kept at OpenAI. Same privacy
   as today on Chat Completions.
5. **Model lists.** Add the documented IDs to the OpenAI chat, image-analysis, summarization and
   Sentinel lists where they fit. The recommended default stays `gpt-5`.

## Paths to cover

Chat (streamed and schema-first), voice (streamed TTS, turn-start acknowledgement), the
PIN-held action flow, image analysis, summarization, Sentinel triage and discovery when they run
on OpenAI, the fallback chain, token counting.

## Tests

- Unit: routing predicate, effort mapping per model, normalization of Responses-shaped messages,
  history cleaning before a fallback to another provider, `store=False` in the request.
- A recorded streaming fixture for a Responses tool turn.
- Live smoke test with Lindo's OpenAI key (never read from `core.config_entries`): a GPT-6 tool
  turn, a streamed voice reply, a PIN-held action, a mid-conversation fallback to Ollama, and an
  old Chat Completions thread continued on GPT-6.

## Out of scope

`previous_response_id` (the checkpointer owns history), OpenAI built-in tools, `reasoning.mode`,
changing the recommended default model.

## Delivery

One PR, gstack review (model-provider plumbing), CHANGELOG `## [Unreleased]` entry, README model
list, then a pre-release offered on Discussion #928.
