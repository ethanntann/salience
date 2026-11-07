// Salience's relay: testers describe their games with your Fireworks key
// without ever holding it. Each install has its own token (from an invite
// code), a monthly allowance of match time, and a share of one limit on the
// key. The key itself is a Worker secret and never leaves Cloudflare.
//
// Frames pass through on their way to Fireworks and are never stored.

import { Gate } from "./gate.js";
import { D1Store } from "./store.js";
import { allowanceSeconds, checkRequest, costUsd, monthKey, resetsAt, usage } from "./rules.js";

export { Gate };

const FIREWORKS = "https://api.fireworks.ai/inference/v1";

function json(body, status = 200, headers = {}) {
  return new Response(JSON.stringify(body), {
    status, headers: { "content-type": "application/json", ...headers } });
}

function error(status, code, message, extra = {}, headers = {}) {
  return json({ error: { code, message, ...extra } }, status, headers);
}

async function sha256(text) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function newToken() {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  return "slc_" + btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function bearer(request) {
  const header = request.headers.get("authorization") || "";
  return header.startsWith("Bearer ") ? header.slice(7).trim() : "";
}

// The default gate: the one Durable Object every call passes through.
function durableGate(env) {
  const stub = env.GATE.get(env.GATE.idFromName("global"));
  return {
    async acquire(player) {
      return (await stub.fetch(`https://gate/acquire?player=${encodeURIComponent(player)}`)).json();
    },
    async release(id) {
      await stub.fetch(`https://gate/release?id=${encodeURIComponent(id)}`);
    },
  };
}

export async function handle(request, env, deps = {}) {
  const store = deps.store || new D1Store(env.DB);
  const upstream = deps.fetch || fetch;
  const now = deps.now || new Date();
  const url = new URL(request.url);
  const path = url.pathname;

  if (path === "/v1/register" && request.method === "POST") {
    const body = await request.json().catch(() => ({}));
    const code = String(body.invite || "").trim();
    if (!code || !(await store.redeemInvite(code))) {
      return error(403, "invite_invalid", "That invite code isn't valid or has been used up.");
    }
    const token = newToken();
    await store.addInstall(await sha256(token), code, now.toISOString());
    return json({ token, base_url: `${url.origin}/v1`, model: String(env.MODEL).split(",")[0].trim(),
      ...usage(0, allowanceSeconds(env, "free")), resets_at: resetsAt(now) });
  }

  if (path.startsWith("/admin/")) {
    if (!env.ADMIN_TOKEN || bearer(request) !== env.ADMIN_TOKEN) return error(401, "unauthorized", "Admin token required.");
    if (path === "/admin/invites" && request.method === "POST") {
      const body = await request.json().catch(() => ({}));
      const code = String(body.code || newToken().slice(4, 14));
      await store.addInvite(code, Math.max(1, Number(body.uses) || 1), String(body.note || ""));
      return json({ code, uses: Math.max(1, Number(body.uses) || 1) });
    }
    if (path === "/admin/usage") return json(await store.report(url.searchParams.get("month") || monthKey(now)));
    if (path === "/admin/disable" && request.method === "POST") {
      const body = await request.json().catch(() => ({}));
      const prefix = String(body.id || "");
      if (prefix.length < 8) return error(400, "bad_request", "Give at least 8 characters of the player id.");
      await store.setDisabled(prefix, body.disabled !== false);
      return json({ ok: true });
    }
    return error(404, "not_found", "No such admin route.");
  }

  // Everything below is a player's: their token, hashed, is who they are.
  const token = bearer(request);
  const hash = token ? await sha256(token) : "";
  const install = hash ? await store.findInstall(hash) : null;
  if (!install) return error(401, "token_invalid", "This install isn't registered. Enter an invite code in Salience.");
  if (install.disabled) return error(403, "disabled", "This install has been switched off.");
  const month = monthKey(now);
  const allowance = allowanceSeconds(env, install.tier);
  const used = await store.monthUsage(hash, month);

  if (path === "/v1/usage") {
    return json({ ...usage(used.calls, allowance), tier: install.tier, resets_at: resetsAt(now) });
  }

  if (path === "/v1/models") {
    return json({ object: "list", data: String(env.MODEL).split(",").map((id) => ({ id: id.trim(), object: "model" })) });
  }

  if (path === "/v1/chat/completions" && request.method === "POST") {
    if (Number(request.headers.get("content-length") || 0) > Number(env.MAX_BODY_BYTES || 12e6)) {
      return error(413, "too_large", "Request too large.");
    }
    const body = await request.json().catch(() => null);
    const problem = checkRequest(env, body);
    if (problem) return error(400, "bad_request", problem);
    if (usage(used.calls, allowance).left_seconds <= 0) {
      return error(402, "allowance_used", "This month's free match time is used up.",
        { resets_at: resetsAt(now) });
    }
    if ((await store.monthTotal(month)) >= Number(env.MONTHLY_BUDGET_USD || 60)) {
      return error(503, "relay_budget", "Salience's beta has reached its monthly limit.", { resets_at: resetsAt(now) });
    }
    const gate = deps.gate || durableGate(env);
    const slot = await gate.acquire(hash);
    if (!slot.ok) {
      return error(429, "busy", "Lots of players right now.", {}, { "retry-after": String(slot.retry_after) });
    }
    try {
      const reply = await upstream(`${env.FIREWORKS_BASE || FIREWORKS}/chat/completions`, {
        method: "POST",
        headers: { authorization: `Bearer ${env.FIREWORKS_API_KEY}`, "content-type": "application/json",
          "user-agent": "salience-relay/0.1" },
        body: JSON.stringify(body),
      });
      const text = await reply.text();
      if (reply.status === 429) {
        return error(429, "busy", "Fireworks is busy.", {},
          { "retry-after": reply.headers.get("retry-after") || "10" });
      }
      if (!reply.ok) return new Response(text, { status: reply.status, headers: { "content-type": "application/json" } });
      let parsed = null;
      try {
        parsed = JSON.parse(text);
      } catch {
        // Answered but unreadable: still charged, it was paid for.
      }
      await store.charge(hash, month, costUsd(env, parsed));
      return new Response(text, { status: 200, headers: { "content-type": "application/json" } });
    } finally {
      await gate.release(slot.id);
    }
  }

  return error(404, "not_found", "No such route.");
}

export default {
  fetch(request, env) {
    return handle(request, env);
  },
};
