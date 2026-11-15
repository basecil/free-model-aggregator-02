# free-model-aggregator-02

One server that makes OpenRouter's free models a single endpoint: an OpenAI
wire and an Anthropic wire, JSON and streaming, with a real free model behind
every answer. Entry for the make.fun bounty of the same name.

## How it works

- The trial brief (`$MAKEFUN_PROMPT_FILE`) lists the free models with their
  published ratings and capabilities. Each request is routed to the
  highest-rated free model that meets its own requirements (JSON mode, tools,
  vision), read off the request body itself.
- Every model is dialled on Chat Completions through the grader's gateway.
  Anthropic requests are translated both ways; the gateway's answer is
  translated back, with the dialled model id put back on every response and
  every streamed chunk (the provider spells it without the `openrouter/`
  prefix).
- Failover: a refusal (429/5xx), a timeout, an unparseable answer, or an
  answer that ignores the prompt's token fails over to the next-best model
  (3 attempts, fixed 12s + 8s + 7s timeouts — inside the client's 30s).
  Models that refused earlier in the trial sort last. Recovered refusals are
  reported in the `x-makefun-upstream-failures` header, in refusal order.
- Streams are synthesized from the buffered upstream answer, never relayed,
  so the entry verifies the answer carries the prompt's token before sending
  anything. The first successful answer is kept as a fallback: served when no
  model echoes the token, so its wire checks still count.
- Deterministic for the replay check: no clock and no randomness in anything
  sent, fixed per-attempt timeouts instead of a wall-clock deadline, response
  ids derived from the upstream answer.

## Layout

- `main.ts` — the server: routes, routing, failover, register, `/work` poll.
- `lib/brief.ts` — the brief's two documents, the catalog, the ranking.
- `lib/wire.ts` — Anthropic ↔ Chat Completions translation, the model field,
  the SSE writers.
- `lib/gateway.ts` — one Chat Completions dial per attempt.

Standard library only — the trial sandbox installs nothing.

## Run

```sh
node main.ts --list   # print the registered catalog, exit 0
node main.ts          # serve a trial (needs MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE, OPENAI_BASE_URL)
```

The entry needs an OpenRouter API key (on an account with ≥10 credits
purchased) pushed to the competition as `OPENROUTER_API_KEY` before submit;
without it every model call is refused. The key never enters the sandbox and
is never committed here.
