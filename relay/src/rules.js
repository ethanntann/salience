// What a tester may ask for and how much of their allowance it uses. Pure
// functions: the Worker (`index.js`) applies them, the tests check them.

// One description is one window of a match: 8 seconds of play. Counting calls,
// not a length the app reports, means the count cannot be talked down.
export const SECONDS_PER_CALL = 8;

export function monthKey(now = new Date()) {
  return now.toISOString().slice(0, 7); // "2026-11", in UTC
}

export function resetsAt(now = new Date()) {
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() + 1, 1)).toISOString();
}

// The allowance for a tier, in seconds of matches.
export function allowanceSeconds(env, tier) {
  const hours = Number(tier === "pro" ? env.PRO_HOURS : env.FREE_HOURS);
  return Math.round((Number.isFinite(hours) ? hours : 5) * 3600);
}

export function usage(calls, allowance) {
  const used = calls * SECONDS_PER_CALL;
  return { used_seconds: used, allowance_seconds: allowance, left_seconds: Math.max(0, allowance - used) };
}

// Estimated cost of one call from what Fireworks reports it used. An estimate
// for the server's own monthly cap, never the invoice.
export function costUsd(env, reply) {
  const u = (reply && reply.usage) || {};
  const input = Number(u.prompt_tokens) || 0;
  const output = Number(u.completion_tokens) || 0;
  return (input * Number(env.PRICE_IN_PER_M || 0.3) + output * Number(env.PRICE_OUT_PER_M || 1.2)) / 1e6;
}

// A request the relay will forward, or why not. Your key may only describe
// gameplay with the one model Salience uses, never be used for anything else.
export function checkRequest(env, body) {
  if (!body || typeof body !== "object") return "not a JSON request";
  const allowed = String(env.MODEL || "").split(",").map((m) => m.trim()).filter(Boolean);
  if (!allowed.includes(body.model)) return `model ${body.model} is not offered`;
  if (body.stream) return "streaming is not offered";
  if (!Array.isArray(body.messages) || body.messages.length === 0 || body.messages.length > 4) {
    return "unexpected messages";
  }
  const maxTokens = Number(body.max_tokens);
  if (!Number.isFinite(maxTokens) || maxTokens < 1 || maxTokens > Number(env.MAX_OUTPUT_TOKENS || 1500)) {
    return "max_tokens out of range";
  }
  if (body.n !== undefined && body.n !== 1) return "one answer per request";
  let images = 0;
  for (const message of body.messages) {
    if (Array.isArray(message.content)) {
      images += message.content.filter((part) => part && part.type === "image_url").length;
    }
  }
  if (images > Number(env.MAX_IMAGES || 16)) return "too many images";
  return null;
}
