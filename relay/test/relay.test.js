import { test } from "node:test";
import assert from "node:assert/strict";

import { handle } from "../src/index.js";
import { Limits } from "../src/gate.js";
import { MemoryStore } from "../src/store.js";
import { checkRequest, monthKey, resetsAt } from "../src/rules.js";

const MODEL = "accounts/fireworks/models/deepseek-v4p1-flash";
const env = {
  MODEL, FREE_HOURS: "5", PRO_HOURS: "30", MONTHLY_BUDGET_USD: "60",
  PRICE_IN_PER_M: "0.30", PRICE_OUT_PER_M: "1.20", FIREWORKS_API_KEY: "fw_secret", ADMIN_TOKEN: "admin",
};
const now = new Date("2026-11-06T08:00:00Z");
const FRAME = "data:image/jpeg;base64,/9j/AA==";

function describeBody(extra = {}) {
  return { model: MODEL, temperature: 0, max_tokens: 600,
    messages: [{ role: "system", content: "x" },
      { role: "user", content: [{ type: "text", text: "describe" }, { type: "image_url", image_url: { url: FRAME } }] }],
    ...extra };
}

function fireworks(status = 200, headers = {}) {
  const calls = [];
  const fn = async (url, init) => {
    calls.push({ url, init });
    const body = status === 200
      ? { choices: [{ message: { content: "{}" } }], usage: { prompt_tokens: 8484, completion_tokens: 159 } }
      : { error: "busy" };
    return new Response(JSON.stringify(body), { status, headers });
  };
  fn.calls = calls;
  return fn;
}

function openGate() {
  const released = [];
  return { released, acquire: async () => ({ ok: true, id: "lease" }), release: async (id) => released.push(id) };
}

async function call(path, { method = "GET", token, body, deps }) {
  const headers = { "content-type": "application/json" };
  if (token) headers.authorization = `Bearer ${token}`;
  const request = new Request(`https://relay.test${path}`, { method, headers, body: body && JSON.stringify(body) });
  const response = await handle(request, env, { now, ...deps });
  return { status: response.status, headers: response.headers, body: await response.json() };
}

async function registered(store, deps = {}) {
  await store.addInvite("BETA1", 1, "friend");
  const reply = await call("/v1/register", { method: "POST", body: { invite: "BETA1" }, deps: { store, ...deps } });
  return reply.body.token;
}

test("an invite code registers one install, once", async () => {
  const store = new MemoryStore();
  const token = await registered(store);
  assert.match(token, /^slc_/);
  const again = await call("/v1/register", { method: "POST", body: { invite: "BETA1" }, deps: { store } });
  assert.equal(again.status, 403);
  assert.ok(![...store.installs.keys()].includes(token), "only the token's hash is kept");
});

test("a description is forwarded with the key the tester never sees, and charged", async () => {
  const store = new MemoryStore();
  const token = await registered(store);
  const upstream = fireworks();
  const gate = openGate();
  const reply = await call("/v1/chat/completions", { method: "POST", token, body: describeBody(),
    deps: { store, fetch: upstream, gate } });
  assert.equal(reply.status, 200);
  assert.equal(upstream.calls[0].init.headers.authorization, "Bearer fw_secret");
  assert.deepEqual(gate.released, ["lease"]);
  const usage = await call("/v1/usage", { token, deps: { store } });
  assert.equal(usage.body.used_seconds, 8);
  assert.equal(usage.body.allowance_seconds, 5 * 3600);
  assert.ok(Math.abs((await store.monthTotal(monthKey(now))) - 0.002736) < 1e-6);
});

test("no token, a wrong token, or a switched-off install is refused", async () => {
  const store = new MemoryStore();
  const token = await registered(store);
  const deps = { store, fetch: fireworks(), gate: openGate() };
  assert.equal((await call("/v1/chat/completions", { method: "POST", body: describeBody(), deps })).status, 401);
  assert.equal((await call("/v1/chat/completions", { method: "POST", token: "slc_wrong", body: describeBody(), deps })).status, 401);
  const id = [...store.installs.keys()][0].slice(0, 12);
  await call("/admin/disable", { method: "POST", token: "admin", body: { id }, deps });
  assert.equal((await call("/v1/chat/completions", { method: "POST", token, body: describeBody(), deps })).status, 403);
});

test("the key describes gameplay with Salience's model and nothing else", () => {
  assert.equal(checkRequest(env, describeBody()), null);
  assert.match(checkRequest(env, describeBody({ model: "accounts/fireworks/models/kimi-k3" })), /not offered/);
  assert.match(checkRequest(env, describeBody({ stream: true })), /streaming/);
  assert.match(checkRequest(env, describeBody({ max_tokens: 100000 })), /max_tokens/);
  const many = describeBody();
  many.messages[1].content = Array(40).fill({ type: "image_url", image_url: { url: FRAME } });
  assert.match(checkRequest(env, many), /images/);
});

test("a used-up allowance waits for next month, without calling Fireworks", async () => {
  const store = new MemoryStore();
  const token = await registered(store);
  const hash = [...store.installs.keys()][0];
  store.usageRows.set(`${hash}|${monthKey(now)}`, { calls: (5 * 3600) / 8, cost_usd: 6 });
  const upstream = fireworks();
  const reply = await call("/v1/chat/completions", { method: "POST", token, body: describeBody(),
    deps: { store, fetch: upstream, gate: openGate() } });
  assert.equal(reply.status, 402);
  assert.equal(reply.body.error.resets_at, resetsAt(now));
  assert.equal(upstream.calls.length, 0);
});

test("the relay stops at its own monthly budget, whatever Fireworks allows", async () => {
  const store = new MemoryStore();
  const token = await registered(store);
  store.totals.set(monthKey(now), { calls: 1, cost_usd: 60 });
  const reply = await call("/v1/chat/completions", { method: "POST", token, body: describeBody(),
    deps: { store, fetch: fireworks(), gate: openGate() } });
  assert.equal(reply.status, 503);
});

test("busy is a polite wait, from the gate or from Fireworks, and is never charged", async () => {
  const store = new MemoryStore();
  const token = await registered(store);
  const full = { acquire: async () => ({ ok: false, retry_after: 3 }), release: async () => {} };
  const queued = await call("/v1/chat/completions", { method: "POST", token, body: describeBody(),
    deps: { store, fetch: fireworks(), gate: full } });
  assert.equal(queued.status, 429);
  assert.equal(queued.headers.get("retry-after"), "3");
  const gate = openGate();
  const upstreamBusy = await call("/v1/chat/completions", { method: "POST", token, body: describeBody(),
    deps: { store, fetch: fireworks(429, { "retry-after": "7" }), gate } });
  assert.equal(upstreamBusy.status, 429);
  assert.equal(upstreamBusy.headers.get("retry-after"), "7");
  assert.deepEqual(gate.released, ["lease"]);
  assert.equal((await store.monthUsage([...store.installs.keys()][0], monthKey(now))).calls, 0);
});

test("one shared limit: in flight, per minute, and per player", () => {
  const limits = new Limits({ maxInFlight: 3, perMinute: 4, perPlayer: 2 });
  const t = 1_000_000;
  const a1 = limits.acquire("a", t);
  const a2 = limits.acquire("a", t);
  assert.ok(a1.ok && a2.ok);
  assert.equal(limits.acquire("a", t).ok, false, "a player's own share is full");
  assert.ok(limits.acquire("b", t).ok, "another player still gets in");
  assert.equal(limits.acquire("c", t).ok, false, "everyone's in-flight share is full");
  limits.release(a1.id);
  assert.ok(limits.acquire("c", t).ok);
  limits.release(a2.id);
  const minute = limits.acquire("d", t);
  assert.equal(minute.ok, false, "four started this minute");
  assert.ok(minute.retry_after >= 1);
  assert.ok(limits.acquire("d", t + 60_000).ok, "a new minute");
  assert.ok(limits.acquire("e", t + 60_000 + 200_000).ok, "a lost release expires");
});

test("admin routes need the admin token", async () => {
  const store = new MemoryStore();
  assert.equal((await call("/admin/usage", { deps: { store } })).status, 401);
  const made = await call("/admin/invites", { method: "POST", token: "admin", body: { uses: 3, note: "discord" }, deps: { store } });
  assert.equal(made.body.uses, 3);
  assert.equal(store.invites.get(made.body.code).uses_left, 3);
});

test("only Salience's own request shape: no other fields, links or walls of text", () => {
  assert.match(checkRequest(env, describeBody({ tools: [] })), /unexpected fields: tools/);
  const link = describeBody();
  link.messages[1].content[1] = { type: "image_url", image_url: { url: "https://example.com/a.jpg" } };
  assert.match(checkRequest(env, link), /text and gameplay frames/);
  const wall = describeBody();
  wall.messages[1].content[0].text = "x".repeat(40001);
  assert.match(checkRequest(env, wall), /too much text/);
  assert.match(checkRequest(env, describeBody({ messages: [{ role: "assistant", content: "hi" }] })), /unexpected messages/);
});

test("a player's dollars run out before their hours if their calls are costly", async () => {
  const store = new MemoryStore();
  const token = await registered(store);
  const hash = [...store.installs.keys()][0];
  store.usageRows.set(`${hash}|${monthKey(now)}`, { calls: 10, cost_usd: 9 }); // 5 h x $1.80
  const upstream = fireworks();
  const reply = await call("/v1/chat/completions", { method: "POST", token, body: describeBody(),
    deps: { store, fetch: upstream, gate: openGate() } });
  assert.equal(reply.status, 402);
  assert.equal(upstream.calls.length, 0);
});

test("calls in flight count against the allowance", async () => {
  const store = new MemoryStore();
  const token = await registered(store);
  const hash = [...store.installs.keys()][0];
  store.usageRows.set(`${hash}|${monthKey(now)}`, { calls: (5 * 3600) / 8 - 3, cost_usd: 0 });
  const released = [];
  const busy = { acquire: async () => ({ ok: true, id: "l", mine: 4 }), release: async (id) => released.push(id) };
  const upstream = fireworks();
  const reply = await call("/v1/chat/completions", { method: "POST", token, body: describeBody(),
    deps: { store, fetch: upstream, gate: busy } });
  assert.equal(reply.status, 402);
  assert.deepEqual(released, ["l"]);
  assert.equal(upstream.calls.length, 0);
});

test("Fireworks' own error text stays in the relay", async () => {
  const store = new MemoryStore();
  const token = await registered(store);
  const reply = await call("/v1/chat/completions", { method: "POST", token, body: describeBody(),
    deps: { store, fetch: fireworks(400), gate: openGate() } });
  assert.equal(reply.status, 400);
  assert.equal(reply.body.error.code, "upstream_error");
  assert.ok(!JSON.stringify(reply.body).includes("busy"));
});

test("disabling takes a hex id, never a wildcard", async () => {
  const store = new MemoryStore();
  await registered(store);
  const reply = await call("/admin/disable", { method: "POST", token: "admin", body: { id: "%%%%%%%%" }, deps: { store } });
  assert.equal(reply.status, 400);
  assert.equal([...store.installs.values()][0].disabled, 0);
  assert.equal((await call("/admin/usage", { token: "admin-ish", deps: { store } })).status, 401);
});
