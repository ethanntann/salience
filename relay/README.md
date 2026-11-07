# Salience relay

Lets beta testers describe their games with **your** Fireworks key without
ever holding it. It runs on Cloudflare Workers:

```
Salience (tester's PC) ──invite token──▶ this relay ──your key──▶ Fireworks
```

- Each install redeems an **invite code** for its own token; only a hash of
  it is stored. Switch off an install any time.
- **5 hours of matches a month** free (each description is 8 s of a match),
  counted per install, reset on the 1st (UTC). Past it, Salience keeps
  recording and makes those highlights when the month resets.
- One **shared limit** on the key: 300 calls in flight, 15,000 a minute (75%
  of the account's 20,000), 16 per player, so nobody starves anyone else.
  Busy answers are a polite "retry in N seconds".
- The relay **stops itself at $60 a month** of estimated spend, whatever
  Fireworks would allow. Set a spend limit on the Fireworks account as well.
- It only forwards descriptions with Salience's model: your key can't be
  used for anything else. Frames pass through and are never stored.

Settings are the `[vars]` in `wrangler.toml`.

## Deploy (about 10 minutes)

You need a free Cloudflare account and Node.js.

```powershell
cd relay
npx.cmd wrangler login
npx.cmd wrangler d1 create salience-relay        # paste the database_id it prints into wrangler.toml
npx.cmd wrangler d1 execute salience-relay --remote --file=schema.sql
npx.cmd wrangler secret put FIREWORKS_API_KEY    # paste your key; it's stored encrypted at Cloudflare
npx.cmd wrangler secret put ADMIN_TOKEN          # any long random string, for the admin routes
npx.cmd wrangler deploy                          # prints https://salience-relay.<you>.workers.dev
```

Then set that address as `DEFAULT_RELAY_URL` in `pipeline_v2/capture/app.py`
so the app knows where invite codes go.

## Running the beta

Make an invite (one code per tester, or one with several uses):

```powershell
curl -X POST https://salience-relay.<you>.workers.dev/admin/invites `
  -H "Authorization: Bearer <ADMIN_TOKEN>" -H "Content-Type: application/json" `
  -d '{"uses": 1, "note": "Alex"}'
```

See who used what this month, and switch one off:

```powershell
curl https://salience-relay.<you>.workers.dev/admin/usage -H "Authorization: Bearer <ADMIN_TOKEN>"
curl -X POST https://salience-relay.<you>.workers.dev/admin/disable `
  -H "Authorization: Bearer <ADMIN_TOKEN>" -H "Content-Type: application/json" -d '{"id": "<first 12 of the id>"}'
```

## Try it on this PC

`node dev.js` runs the relay locally with invite code `DEV`. Without
`FIREWORKS_API_KEY` set, a stand-in answers instead of Fireworks, so nothing
is spent. Point the app at it with `SALIENCE_RELAY_URL=http://127.0.0.1:8787`.

Tests: `npm test` (Node's built-in runner, nothing to install).
