// free-model-aggregator-02: one server, two client wires, one real free
// model behind every answer.
//
// The brief at $MAKEFUN_PROMPT_FILE lists the free models (ratings,
// capabilities) and the six prompts. Each request is routed to the
// highest-rated free model that meets its own requirements (JSON mode, tools,
// vision); a refusal, a timeout, or an answer that ignores the prompt's token
// fails over to the next-best model. Recovered refusals are reported in the
// x-makefun-upstream-failures header, in refusal order.
//
//   node main.ts --list   prints the registered catalog and exits 0.
//   node main.ts          serves the trial, registers, polls /work, exits 0.
//
// Standard library only: the trial sandbox installs nothing.
//
// REPLAY NOTE. The final replays each trial against the recorded gateway
// calls and the state export must match byte for byte, so the dial sequence
// is a function of the answers alone, never of the wall clock: a fixed
// number of attempts with fixed per-attempt timeouts (12s + 8s + 7s = 27s
// worst case, inside the client's 30s per-prompt timeout), no per-prompt
// deadline, no randomness, no clock in anything sent. A dial that hangs past
// its timeout is the only timing-dependent branch, and it only fires for a
// model that was not going to answer in time anyway.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  freeModels,
  isObject,
  rankedCandidates,
  readBrief,
  type Brief,
  type Catalog,
  type JsonObject,
} from './lib/brief.ts';
import { createGateway } from './lib/gateway.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  estimateTokens,
  needsFromBody,
  openAIError,
  openAIModelList,
  parseUpstreamAnswer,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  textContent,
  type ListedModel,
  type UpstreamAnswer,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** Most models tried for one prompt. */
const MAX_ATTEMPTS = 3;
/** One dial's timeout, by attempt: 12s + 8s + 7s = 27s worst case, inside the client's 30s. */
const ATTEMPT_TIMEOUTS_MS = [12000, 8000, 7000];
/** Most trail rows reported on one answer. */
const MAX_TRAIL_ROWS = 8;
/** The gateway's per-call output cap. */
const MAX_OUTPUT_TOKENS = 8192;
/** The output cap used when a request names none. */
const DEFAULT_OUTPUT_TOKENS = 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: Catalog = buildCatalog(brief);
const gateway = createGateway(GATEWAY_URL, GATEWAY_KEY);

/** Models that refused or timed out earlier this trial: tried last, never first. */
const coolingDown = new Set<string>();

/** One recovered refusal, as the trail header reports it. */
interface TrailRow {
  status: number;
  code: string;
  model: string | null;
}

/** A prompt served: the answer, the model that really gave it, and whether it carries the token. */
interface ServeOutcome {
  answer: UpstreamAnswer;
  model: string;
  tokenOk: boolean;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** The free models the brief lists, each once, as GET /v1/models names them. */
function listedModels(): ListedModel[] {
  const seen = new Set<string>();
  const out: ListedModel[] = [];
  for (const model of freeModels(brief)) {
    if (seen.has(model.id)) continue;
    seen.add(model.id);
    out.push({ id: model.id, provider: model.provider, displayName: model.displayName });
  }
  return out;
}

/** A prompt token (e.g. ZEBRA-4721) in the request's messages, or null when it carries none. */
const TOKEN_PATTERN = /[A-Z]{2,12}-\d{4}/;

function promptToken(body: JsonObject): string | null {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (!isObject(message)) continue;
    const match = TOKEN_PATTERN.exec(textContent(message.content));
    if (match !== null) return match[0];
  }
  return null;
}

/** All the request's text, for the usage estimate when the provider reports none. */
function requestTextOf(body: JsonObject): string {
  const parts: string[] = [];
  const system = textContent(body.system);
  if (system !== '') parts.push(system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (isObject(message)) parts.push(textContent(message.content));
  }
  return parts.join('\n');
}

/** The x-makefun-upstream-failures header value for these rows, or null when there are none. */
function trailHeader(rows: TrailRow[]): string | null {
  if (rows.length === 0) return null;
  const value = JSON.stringify(rows.slice(0, MAX_TRAIL_ROWS));
  if (Buffer.byteLength(value, 'utf8') > 8192) return null;
  return value;
}

/** An OpenAI client body as the upstream body: never streamed, the model set by the dial. */
function openAIUpstreamBody(body: JsonObject): JsonObject {
  const out: JsonObject = { ...body };
  delete out.stream;
  delete out.stream_options;
  return out;
}

/**
 * The usage a response carries: the provider's own numbers when it reported
 * them (a non-empty completion is at least one output token), else a rough
 * estimate from the request and answer text.
 */
function usageOf(answer: UpstreamAnswer, requestText: string): { prompt: number; completion: number } {
  const prompt = answer.usageReported ? Math.max(0, answer.promptTokens) : estimateTokens(requestText);
  const completion = answer.usageReported
    ? answer.text !== ''
      ? Math.max(1, answer.completionTokens)
      : Math.max(0, answer.completionTokens)
    : Math.max(1, estimateTokens(answer.text));
  return { prompt, completion };
}

/**
 * Serve one request: dial the best free model that meets it, and fail over
 * down the ranking on a refusal, a timeout, an unparseable answer, or an
 * answer that ignores the prompt's token. `failures` collects the trail rows
 * in refusal order. Returns the first token-carrying answer, else the first
 * successful answer as a fallback (its wire checks still count), else null.
 */
async function serve(body: JsonObject, wire: 'openai' | 'anthropic', failures: TrailRow[]): Promise<ServeOutcome | null> {
  const candidates = rankedCandidates(brief, needsFromBody(body), coolingDown);
  if (candidates.length === 0) {
    log('no free model in the brief meets this request');
    return null;
  }
  const token = promptToken(body);
  let fallback: ServeOutcome | null = null;
  const attempts = candidates.slice(0, MAX_ATTEMPTS);
  for (let i = 0; i < attempts.length; i++) {
    const model = attempts[i];
    const upstreamBody = wire === 'anthropic' ? anthropicToOpenAI(body, model.id) : openAIUpstreamBody(body);
    if (typeof upstreamBody.max_tokens !== 'number' && typeof upstreamBody.max_completion_tokens !== 'number') {
      upstreamBody.max_tokens = DEFAULT_OUTPUT_TOKENS;
    }
    if (typeof upstreamBody.max_tokens === 'number' && upstreamBody.max_tokens > MAX_OUTPUT_TOKENS) upstreamBody.max_tokens = MAX_OUTPUT_TOKENS;
    if (typeof upstreamBody.max_completion_tokens === 'number' && upstreamBody.max_completion_tokens > MAX_OUTPUT_TOKENS) {
      upstreamBody.max_completion_tokens = MAX_OUTPUT_TOKENS;
    }
    let result;
    try {
      result = await gateway.dial(model.id, upstreamBody, ATTEMPT_TIMEOUTS_MS[i]);
    } catch (err) {
      // A network failure or a timeout: no gateway refusal to report, but the
      // model is cooling down for the rest of the trial.
      log(`dial ${model.id} failed: ${(err as Error).message}`);
      coolingDown.add(model.id);
      continue;
    }
    if (!result.ok) {
      failures.push({ status: result.refusal.status, code: result.refusal.code, model: model.id });
      coolingDown.add(model.id);
      log(`dial ${model.id} refused HTTP ${result.refusal.status} (${result.refusal.code})`);
      continue;
    }
    const answer = parseUpstreamAnswer(result.body);
    if (answer === null) {
      log(`dial ${model.id} answered unparseably`);
      coolingDown.add(model.id);
      continue;
    }
    const outcome: ServeOutcome = { answer, model: model.id, tokenOk: token === null || answer.text.includes(token) };
    if (fallback === null) fallback = outcome;
    if (outcome.tokenOk) return outcome;
    log(`dial ${model.id} answered without the prompt token; trying the next model`);
  }
  return fallback;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readBody(req: IncomingMessage): Promise<JsonObject | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isObject(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, trail: string | null): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
  };
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(status, headers);
  res.end(text);
}

function sendStream(res: ServerResponse, body: string, trail: string | null): void {
  const headers: Record<string, string> = {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  };
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(200, headers);
  res.end(body);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, trail: string | null): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), trail);
  else sendJson(res, status, openAIError(message, type, null), trail);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(listedModels()), null);
  else sendJson(res, 200, openAIModelList(listedModels()), null);
}

/** One prompt answered on either wire: served, or a 502 carrying the trail of what was tried. */
async function answer(res: ServerResponse, wire: 'openai' | 'anthropic', body: JsonObject): Promise<void> {
  const failures: TrailRow[] = [];
  const outcome = await serve(body, wire, failures);
  const trail = trailHeader(failures);
  if (outcome === null) {
    sendError(res, wire, 502, wire === 'anthropic' ? 'api_error' : 'upstream_error', 'no free model served this prompt', trail);
    return;
  }
  const usage = usageOf(outcome.answer, requestTextOf(body));
  const stream = body.stream === true;
  if (wire === 'openai') {
    if (stream) sendStream(res, renderOpenAIStream(outcome.answer, outcome.model, usage.prompt, usage.completion), trail);
    else sendJson(res, 200, renderOpenAIJson(outcome.answer, outcome.model, usage.prompt, usage.completion), trail);
  } else {
    if (stream) sendStream(res, renderAnthropicStream(outcome.answer, outcome.model, usage.prompt, usage.completion), trail);
    else sendJson(res, 200, renderAnthropicJson(outcome.answer, outcome.model, usage.prompt, usage.completion), trail);
  }
  log(
    `served ${wire} ${stream ? 'stream' : 'json'} with ${outcome.model}${outcome.tokenOk ? '' : ' (no token)'}${failures.length > 0 ? ` after ${failures.length} refusal(s)` : ''}`,
  );
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by a real free model. */
async function completions(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readBody(req);
  if (body === null) {
    sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object', null);
    return;
  }
  await answer(res, 'openai', body);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readBody(req);
  if (body === null) {
    sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object', null);
    return;
  }
  await answer(res, 'anthropic', body);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await completions(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`, null);
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isObject(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Register, retrying briefly while the env comes up. */
async function registerWithRetry(port: number): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      await register(port);
      return;
    } catch (err) {
      lastError = err;
      log(`register attempt ${attempt + 1} failed: ${(err as Error).message}`);
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }
  throw lastError;
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function pollWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isObject(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(`brief: ${brief.error}`);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts. A process whose
    // stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (GATEWAY_URL === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer', null);
      }
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (address.port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${address.port} with ${listedModels().length} free models`);
  await registerWithRetry(address.port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
