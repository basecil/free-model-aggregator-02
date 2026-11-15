// The trial brief: the free model list, the prompt manifest, the catalog,
// and the rating-ranked routing table.
//
// The brief at $MAKEFUN_PROMPT_FILE is prose with two JSON documents embedded
// in it, found by their `schema` member under this competition's prefix. The
// prose around them is never parsed.
//
//   - the free model list: one row per model, keyed by the id the gateway
//     routes (`openrouter/<vendor>/<model>:free`), with its published rating
//     in milli-points, what it can do (tools, vision, JSON mode, context
//     length) and its free flag. Only a free model may serve a prompt.
//   - the six prompts: each one's id, its api, whether it streams, its weight
//     and the requirements a model must meet to serve it. The client sends the
//     bodies; the server routes each arrival by the body's own features, so
//     the manifest is a cross-check, not a dependency.
//
// Everything here is a pure function of the brief's bytes: no clock, no
// randomness, so the catalog printed by `--list` and the one POSTed to
// /register are the same document, byte for byte, on a live pass and on its
// replay.

import { readFileSync } from 'node:fs';

/** The public base URL of the provider the free models belong to. */
export const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';
/** The environment variable an entrant's OpenRouter key is pushed under. */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

/** Every schema this competition's brief writes names this prefix. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

export type JsonObject = Record<string, unknown>;

export function isObject(value: unknown): value is JsonObject {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** One free model as the brief lists it. `id` is the id the gateway routes. */
export interface FreeModel {
  id: string;
  provider: string;
  displayName: string;
  /** The published rating in milli-points: the quality column's currency. */
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one request needs: a model without it cannot serve the request. */
export interface PromptNeeds {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

/** One prompt the client will send, as the brief lists it. */
export interface PromptEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: PromptNeeds;
}

/** The brief as this entry reads it. `error` is set when a document was missing. */
export interface Brief {
  models: FreeModel[];
  prompts: PromptEntry[];
  error: string | null;
}

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** What `--list` prints and what the entry POSTs to /register: one document. */
export interface Catalog {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: string;
    models: CatalogModel[];
  }>;
}

/** The index of the `}` closing the object that opens at `open`, or -1. String contents are skipped. */
function matchingBrace(text: string, open: number): number {
  let depth = 0;
  let inString = false;
  for (let i = open; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function schemaDocuments(text: string): JsonObject[] {
  const docs: JsonObject[] = [];
  let at = text.indexOf('{');
  while (at !== -1) {
    const end = matchingBrace(text, at);
    if (end !== -1) {
      try {
        const parsed: unknown = JSON.parse(text.slice(at, end + 1));
        if (isObject(parsed) && typeof parsed.schema === 'string') docs.push(parsed);
      } catch {
        // Not a standalone object: keep scanning.
      }
    }
    at = text.indexOf('{', at + 1);
  }
  return docs;
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function asNumberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function asBoolOrNull(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the provider
 * that names it: a row giving the bare OpenRouter id (`vendor/model:free`)
 * and naming `openrouter` is routed as the gateway spells it
 * (`openrouter/vendor/model:free`). A row is free unless it says otherwise;
 * a paid row is never routed.
 */
function parseModel(value: unknown): FreeModel | null {
  if (!isObject(value)) return null;
  const raw = asString(value.id) !== '' ? asString(value.id) : asString(value.model);
  if (raw === '') return null;
  const provider = asString(value.provider) !== '' ? asString(value.provider) : asString(value.wire) !== '' ? asString(value.wire) : 'openrouter';
  const id = provider === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider,
    displayName: asString(value.display_name) !== '' ? asString(value.display_name) : raw,
    ratingMilli: asNumberOrNull(value.rating_milli) ?? 0,
    contextLength: asNumberOrNull(value.context_length),
    tools: asBoolOrNull(value.tools),
    vision: asBoolOrNull(value.vision),
    json: asBoolOrNull(value.json),
    free: value.free !== false,
  };
}

function parsePrompt(value: unknown): PromptEntry | null {
  if (!isObject(value)) return null;
  const id = asString(value.id);
  if (id === '') return null;
  const req = isObject(value.requires) ? value.requires : {};
  const needs: PromptNeeds = {};
  if (req.tools === true) needs.tools = true;
  if (req.vision === true) needs.vision = true;
  if (req.json === true) needs.json = true;
  if (typeof req.min_context === 'number' && Number.isFinite(req.min_context)) needs.minContext = req.min_context;
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: needs,
  };
}

/** The first document under this competition's prefix carrying one of `keys` as an array. */
function findArrayDoc(docs: readonly JsonObject[], keys: readonly string[]): unknown[] | null {
  for (const doc of docs) {
    if (!asString(doc.schema).includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) {
      if (Array.isArray(doc[key])) return doc[key] as unknown[];
    }
  }
  return null;
}

/** The brief at `path`, its two documents found under this competition's schema prefix. */
export function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read the brief at ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const modelRows = findArrayDoc(docs, ['models', 'ratings']);
  if (modelRows === null) return { models: [], prompts: [], error: 'the brief holds no free-model list document' };
  const models: FreeModel[] = [];
  for (const row of modelRows) {
    const model = parseModel(row);
    if (model !== null) models.push(model);
  }
  const promptRows = findArrayDoc(docs, ['prompts', 'manifest']);
  const prompts: PromptEntry[] = [];
  if (promptRows !== null) {
    for (const row of promptRows) {
      const prompt = parsePrompt(row);
      if (prompt !== null) prompts.push(prompt);
    }
  }
  return { models, prompts, error: promptRows === null ? 'the brief holds no prompts document' : null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
export function freeModels(brief: Brief): FreeModel[] {
  return brief.models.filter((model) => model.free);
}

/** Whether a model's capabilities cover one request's needs. An unknown capability never covers one. */
export function meetsNeeds(needs: PromptNeeds, model: FreeModel): boolean {
  if (needs.tools === true && model.tools !== true) return false;
  if (needs.vision === true && model.vision !== true) return false;
  if (needs.json === true && model.json !== true) return false;
  if (needs.minContext !== undefined && (model.contextLength === null || model.contextLength < needs.minContext)) return false;
  return true;
}

/**
 * The free models that can serve a request, best first: highest published
 * rating, ties broken by id so the order is the same on every pass. Models in
 * `coolingDown` — refused or timed out earlier this trial — sort after the
 * rest; they are still tried when nothing else is left.
 */
export function rankedCandidates(brief: Brief, needs: PromptNeeds, coolingDown: ReadonlySet<string>): FreeModel[] {
  const eligible = freeModels(brief).filter((model) => meetsNeeds(needs, model));
  eligible.sort((a, b) => {
    const aCool = coolingDown.has(a.id) ? 1 : 0;
    const bCool = coolingDown.has(b.id) ? 1 : 0;
    if (aCool !== bCool) return aCool - bCool;
    if (b.ratingMilli !== a.ratingMilli) return b.ratingMilli - a.ratingMilli;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  return eligible;
}

/**
 * The catalog the brief's free model list makes: one provider, the models it
 * serves, each once.
 *
 * The base_url is the provider's public endpoint, never the gateway's
 * trial-local address: the gateway's origin differs between a live pass and
 * its replay, and the catalog is part of the state export a replay must
 * reproduce byte for byte.
 */
export function buildCatalog(brief: Brief): Catalog {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of freeModels(brief)) {
    if (seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.contextLength });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_BASE_URL,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}
