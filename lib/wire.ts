// The two client wires and the one upstream wire.
//
// The client speaks OpenAI (POST /v1/chat/completions) or Anthropic (POST
// /v1/messages), JSON or SSE. Upstream is always Chat Completions, so
// Anthropic requests are translated both ways and the gateway's answer is
// translated back. Every response names the model this entry really dialled,
// on every JSON body and every streamed chunk: the provider's own answer
// spells it without the gateway's `openrouter/` prefix, and these functions
// put the dialled id back.
//
// Streams are synthesized from the buffered upstream answer, never relayed:
// the entry verifies the answer carries the prompt's token before it sends
// anything, so a model that ignored the instruction costs one more dial,
// not the prompt.
//
// Everything here is a pure function of its arguments: fixed timestamps,
// deterministic ids, no clock, no randomness, so the same body and the same
// gateway answer give the same bytes on a replay.

import { createHash } from 'node:crypto';
import { isObject, type JsonObject, type PromptNeeds } from './brief.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ListedModel {
  id: string;
  provider: string;
  displayName: string;
}

/** A tool call in a Chat Completions answer. */
export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

/** A Chat Completions answer, normalized for rendering on either wire. */
export interface UpstreamAnswer {
  /** The provider's own response id ('' when absent). */
  upstreamId: string;
  /** The provider's created epoch second (0 when absent). */
  created: number;
  text: string;
  toolCalls: ToolCall[];
  finishReason: string;
  promptTokens: number;
  completionTokens: number;
  /** Whether the provider reported usage at all. */
  usageReported: boolean;
}

/** A fixed epoch second for every listed model's `created`: nothing in a reply reads the clock. */
const LIST_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in a reply reads the clock. */
const LIST_CREATED_AT = '2026-01-01T00:00:00Z';

/** A deterministic response id: the same upstream answer gives the same id on a replay. */
export function answerId(prefix: string, upstreamId: string, model: string, text: string): string {
  const hash = createHash('sha256').update(`${upstreamId}|${model}|${text}`).digest('hex');
  return `${prefix}${hash.slice(0, 24)}`;
}

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ListedModel[]): JsonObject {
  return {
    object: 'list',
    data: models.map((model) => ({ id: model.id, object: 'model', created: LIST_CREATED, owned_by: model.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ListedModel[]): JsonObject {
  return {
    data: models.map((model) => ({ type: 'model', id: model.id, display_name: model.displayName, created_at: LIST_CREATED_AT })),
    has_more: false,
  };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): JsonObject {
  return { error: { message, type, code } };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): JsonObject {
  return { type: 'error', error: { type, message } };
}

/** The text of a string content or of a block array's text blocks; '' when there is none. */
export function textContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isObject(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** Whether a message's content carries an image part, in either wire's spelling. */
function hasImage(content: unknown): boolean {
  if (!Array.isArray(content)) return false;
  return content.some((block) => isObject(block) && (block.type === 'image' || block.type === 'image_url'));
}

/** The requirements a request body declares for itself: tools, JSON mode, vision. */
export function needsFromBody(body: JsonObject): PromptNeeds {
  const needs: PromptNeeds = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) needs.tools = true;
  const format = isObject(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') needs.json = true;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (isObject(message) && hasImage(message.content)) {
      needs.vision = true;
      break;
    }
  }
  return needs;
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function anthropicPart(block: JsonObject): JsonObject | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isObject(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textContent(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as a Chat Completions request naming `model`:
 * the system prompt becomes a system message, image blocks become image_url
 * parts, tool_use blocks become tool_calls, tool_result blocks become tool
 * messages, tools and tool_choice are rewritten, and the output cap,
 * temperature, top_p and stop sequences pass through. Never streamed: the
 * entry always dials non-streaming and synthesizes SSE itself.
 */
export function anthropicToOpenAI(body: JsonObject, model: string): JsonObject {
  const messages: JsonObject[] = [];
  const system = textContent(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isObject(raw)) continue;
    const isAssistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: isAssistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (isAssistant) {
      const toolCalls: JsonObject[] = [];
      for (const block of content) {
        if (!isObject(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isObject(block.input) ? block.input : {}),
          },
        });
      }
      const text = textContent(content);
      const message: JsonObject = { role: 'assistant', content: text !== '' ? text : null };
      if (toolCalls.length > 0) message.tool_calls = toolCalls;
      messages.push(message);
      continue;
    }
    const parts: JsonObject[] = [];
    for (const block of content) {
      if (!isObject(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = anthropicPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: JsonObject = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools: JsonObject[] = [];
    for (const raw of body.tools) {
      if (!isObject(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isObject(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      });
    }
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isObject(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  return out;
}

function wholeOr(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? Math.trunc(value) : fallback;
}

/** A Chat Completions JSON body as the normalized answer this entry renders; null when it has no choice. */
export function parseUpstreamAnswer(body: unknown): UpstreamAnswer | null {
  if (!isObject(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = choices.length > 0 && isObject(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isObject(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textContent(message.content);
  const toolCalls: ToolCall[] = [];
  const rawCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of rawCalls) {
    if (!isObject(call) || !isObject(call.function)) continue;
    toolCalls.push({
      id: typeof call.id === 'string' ? call.id : '',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      arguments: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
    });
  }
  const usage = isObject(body.usage) ? body.usage : null;
  return {
    upstreamId: typeof body.id === 'string' ? body.id : '',
    created: wholeOr(body.created, 0),
    text,
    toolCalls,
    finishReason: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    promptTokens: usage !== null ? wholeOr(usage.prompt_tokens, 0) : 0,
    completionTokens: usage !== null ? wholeOr(usage.completion_tokens, 0) : 0,
    usageReported: usage !== null,
  };
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function anthropicStopReason(finish: string): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** A rough token estimate (4 chars per token), used only when the provider reports no usage. */
export function estimateTokens(text: string): number {
  return Math.max(0, Math.ceil(text.length / 4));
}

function toolArguments(value: string): JsonObject {
  if (value.trim() === '') return {};
  try {
    const parsed: unknown = JSON.parse(value);
    return isObject(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

/** A buffered answer as a Chat Completions object. `model` is the id this entry dialled. */
export function renderOpenAIJson(answer: UpstreamAnswer, model: string, promptTokens: number, completionTokens: number): JsonObject {
  const message: JsonObject = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call) => ({
      // A provider that sends no id still gets a stable one: an empty tool
      // call id is a wire flaw the client cannot answer to.
      id: call.id !== '' ? call.id : answerId('call_', answer.upstreamId, model, call.name),
      type: 'function',
      function: { name: call.name, arguments: call.arguments },
    }));
  }
  return {
    id: answerId('chatcmpl-', answer.upstreamId, model, answer.text),
    object: 'chat.completion',
    created: answer.created > 0 ? answer.created : LIST_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: answer.finishReason }],
    usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
  };
}

/** A buffered answer as an Anthropic Messages object. `model` is the id this entry dialled. */
export function renderAnthropicJson(answer: UpstreamAnswer, model: string, promptTokens: number, completionTokens: number): JsonObject {
  const content: JsonObject[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({
      type: 'tool_use',
      id: call.id !== '' ? call.id : answerId('toolu_', answer.upstreamId, model, call.name),
      name: call.name,
      input: toolArguments(call.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: answerId('msg_', answer.upstreamId, model, answer.text),
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopReason(answer.finishReason),
    stop_sequence: null,
    usage: { input_tokens: promptTokens, output_tokens: completionTokens },
  };
}

/** Split text into streamable pieces on word boundaries, at most ~48 chars each; joined, they are the text. */
function textPieces(text: string): string[] {
  if (text === '') return [];
  const pieces: string[] = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(start + 48, text.length);
    if (end < text.length) {
      const space = text.lastIndexOf(' ', end);
      if (space > start) end = space + 1;
    }
    pieces.push(text.slice(start, end));
    start = end;
  }
  return pieces;
}

function sseData(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

function sseNamed(event: string, payload: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
}

/**
 * A buffered answer as OpenAI SSE: a role chunk, content deltas, one delta
 * per tool call, exactly one finish chunk carrying the usage, then [DONE]
 * and nothing after it. Content deltas always precede the finish chunk.
 */
export function renderOpenAIStream(answer: UpstreamAnswer, model: string, promptTokens: number, completionTokens: number): string {
  const id = answerId('chatcmpl-', answer.upstreamId, model, answer.text);
  const created = answer.created > 0 ? answer.created : LIST_CREATED;
  const frames: string[] = [];
  frames.push(
    sseData({
      id,
      object: 'chat.completion.chunk',
      created,
      model,
      choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
    }),
  );
  for (const piece of textPieces(answer.text)) {
    frames.push(
      sseData({
        id,
        object: 'chat.completion.chunk',
        created,
        model,
        choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
      }),
    );
  }
  for (const call of answer.toolCalls) {
    frames.push(
      sseData({
        id,
        object: 'chat.completion.chunk',
        created,
        model,
        choices: [
          {
            index: 0,
            delta: { tool_calls: [{ id: call.id !== '' ? call.id : answerId('call_', answer.upstreamId, model, call.name), type: 'function', function: { name: call.name, arguments: call.arguments } }] },
            finish_reason: null,
          },
        ],
      }),
    );
  }
  frames.push(
    sseData({
      id,
      object: 'chat.completion.chunk',
      created,
      model,
      choices: [{ index: 0, delta: {}, finish_reason: answer.finishReason }],
      usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/**
 * A buffered answer as the Anthropic event sequence: message_start (first,
 * carrying the model and the input tokens), one text block, one block per
 * tool call, message_delta (the stop reason and the output tokens, only
 * after every block closed), message_stop (last).
 */
export function renderAnthropicStream(answer: UpstreamAnswer, model: string, promptTokens: number, completionTokens: number): string {
  const id = answerId('msg_', answer.upstreamId, model, answer.text);
  const frames: string[] = [];
  frames.push(
    sseNamed('message_start', {
      type: 'message_start',
      message: {
        id,
        type: 'message',
        role: 'assistant',
        content: [],
        model,
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: promptTokens, output_tokens: 1 },
      },
    }),
  );
  let index = 0;
  frames.push(sseNamed('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
  for (const piece of textPieces(answer.text)) {
    frames.push(sseNamed('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(sseNamed('content_block_stop', { type: 'content_block_stop', index }));
  index += 1;
  for (const call of answer.toolCalls) {
    frames.push(
      sseNamed('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'tool_use',
          id: call.id !== '' ? call.id : answerId('toolu_', answer.upstreamId, model, call.name),
          name: call.name,
          input: {},
        },
      }),
    );
    if (call.arguments !== '') {
      frames.push(sseNamed('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.arguments } }));
    }
    frames.push(sseNamed('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseNamed('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopReason(answer.finishReason), stop_sequence: null },
      usage: { output_tokens: completionTokens },
    }),
  );
  frames.push(sseNamed('message_stop', { type: 'message_stop' }));
  return frames.join('');
}
