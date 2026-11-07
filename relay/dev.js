// The relay on this PC, for trying the app against it: `node dev.js`.
// Its records live in memory. With FIREWORKS_API_KEY set it forwards to
// Fireworks (and spends); without, a stand-in answers every description.
import http from "node:http";
import { handle } from "./src/index.js";
import { Limits } from "./src/gate.js";
import { MemoryStore } from "./src/store.js";

const port = Number(process.env.PORT || 8787);
const store = new MemoryStore();
await store.addInvite("DEV", 100, "local");
const limits = new Limits();
const gate = { acquire: async (p) => limits.acquire(p), release: async (id) => limits.release(id) };
const standIn = async () => new Response(JSON.stringify({
  choices: [{ message: { role: "assistant", content: "{\"summary\": \"stand-in\"}" } }],
  usage: { prompt_tokens: 8484, completion_tokens: 159 } }), { headers: { "content-type": "application/json" } });
const env = { MODEL: "accounts/fireworks/models/deepseek-v4p1-flash", FREE_HOURS: "5", MONTHLY_BUDGET_USD: "60",
  ADMIN_TOKEN: "dev", FIREWORKS_API_KEY: process.env.FIREWORKS_API_KEY || "" };

http.createServer(async (req, res) => {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const body = chunks.length ? Buffer.concat(chunks) : undefined;
  const request = new Request(`http://127.0.0.1:${port}${req.url}`, { method: req.method, headers: req.headers,
    body: req.method === "GET" ? undefined : body });
  const reply = await handle(request, env, { store, gate, fetch: env.FIREWORKS_API_KEY ? fetch : standIn });
  res.writeHead(reply.status, Object.fromEntries(reply.headers));
  res.end(Buffer.from(await reply.arrayBuffer()));
}).listen(port, () => console.log(`relay on http://127.0.0.1:${port} (invite code DEV)`));
