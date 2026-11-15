// The grader's gateway: one Chat Completions dial per attempt.
//
// Every model is dialled on Chat Completions only, non-streaming, with the
// trial's dummy key sent as the vendor SDK sends it; the gateway swaps it
// for the entrant's pushed OpenRouter key. A refusal (4xx/5xx) comes back as
// sent, with its error code, so the caller can record it in the trail header
// and fail over. Network failures and timeouts surface as thrown errors:
// there was no gateway refusal to report.

export interface GatewayRefusal {
  status: number;
  code: string;
  message: string;
}

export type DialResult = { ok: true; status: number; body: unknown } | { ok: false; refusal: GatewayRefusal };

export interface Gateway {
  dial(model: string, body: Record<string, unknown>, timeoutMs: number): Promise<DialResult>;
}

/** The error's code string out of a gateway error body, or the fallback. */
function errorCodeOf(body: unknown, fallback: string): string {
  if (typeof body === 'object' && body !== null) {
    const record = body as Record<string, unknown>;
    const error = record.error;
    if (typeof error === 'object' && error !== null) {
      const errorRecord = error as Record<string, unknown>;
      if (typeof errorRecord.code === 'string' && errorRecord.code !== '') return errorRecord.code;
      if (typeof errorRecord.type === 'string' && errorRecord.type !== '') return errorRecord.type;
    }
    if (typeof record.code === 'string' && record.code !== '') return record.code;
  }
  return fallback;
}

export function createGateway(baseUrl: string, apiKey: string): Gateway {
  const root = baseUrl.replace(/\/+$/, '');
  // OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the trial
  // network); a base that does not already end in `/v1` gets one.
  const url = root === '' ? '' : root.endsWith('/v1') ? `${root}/chat/completions` : `${root}/v1/chat/completions`;
  return {
    async dial(model, body, timeoutMs) {
      if (url === '') throw new Error('OPENAI_BASE_URL is not set');
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: `Bearer ${apiKey}` },
        body: JSON.stringify({ ...body, model }),
        signal: AbortSignal.timeout(timeoutMs),
      });
      const text = await res.text();
      let parsed: unknown = null;
      try {
        parsed = text === '' ? null : JSON.parse(text);
      } catch {
        parsed = null;
      }
      if (!res.ok) {
        return { ok: false, refusal: { status: res.status, code: errorCodeOf(parsed, `http_${res.status}`), message: text.slice(0, 500) } };
      }
      return { ok: true, status: res.status, body: parsed };
    },
  };
}
