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

// What a player may spend a month, in dollars: an hour of matches costs about
// $1.20 (450 calls at $0.0027), so this leaves room. Counting calls alone let a
// call stuffed with text cost 100 times a real one (audit, 2026-11-07).
export function playerBudgetUsd(env, tier) {
  const perHour = Number(env.PLAYER_USD_PER_HOUR || 1.8);
  return (allowanceSeconds(env, tier) / 3600) * perHour;
}

// The fields Salience sends, and only those: anything else is not a description.
const FIELDS = new Set(["model", "temperature", "max_tokens", "messages", "reasoning_effort", "top_p", "n", "stream"]);
const ROLES = new Set(["system", "user"]);

// A request the relay will forward, or why not. Your key may only describe
// gameplay with the one model Salience uses, never be used for anything else.
export function checkRequest(env, body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) return "not a JSON request";
  const extra = Object.keys(body).filter((key) => !FIELDS.has(key));
  if (extra.length) return `unexpected fields: ${extra.join(", ")}`;
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
  let text = 0;
  for (const message of body.messages) {
    if (!message || typeof message !== "object" || !ROLES.has(message.role)) return "unexpected messages";
    const parts = typeof message.content === "string" ? [{ type: "text", text: message.content }] : message.content;
    if (!Array.isArray(parts)) return "unexpected messages";
    for (const part of parts) {
      if (part && part.type === "text" && typeof part.text === "string") {
        text += part.text.length;
      } else if (part && part.type === "image_url" && part.image_url
                 && /^data:image\/(jpeg|png|webp);base64,/.test(String(part.image_url.url))) {
        images += 1; // frames travel inside the request: the relay never fetches a link
      } else {
        return "only text and gameplay frames";
      }
    }
  }
  if (images > Number(env.MAX_IMAGES || 16)) return "too many images";
  // Salience's prompt is about 12,000 characters.
  if (text > Number(env.MAX_TEXT_CHARS || 40000)) return "too much text";
  return null;
}
