// The one shared limit on your Fireworks key: how many calls are in flight
// across every player, how many started this minute, and how many one player
// may have at once, so nobody's session starves anyone else's.

export class Limits {
  constructor({ maxInFlight = 300, perMinute = 15000, perPlayer = 16, leaseMs = 180000 } = {}) {
    Object.assign(this, { maxInFlight, perMinute, perPlayer, leaseMs });
    this.leases = new Map(); // id -> { player, at }
    this.minute = { start: 0, count: 0 };
    this.next = 0;
  }

  _expire(now) {
    // A call whose release never came (a dropped connection) frees its slot.
    for (const [id, lease] of this.leases) if (now - lease.at > this.leaseMs) this.leases.delete(id);
  }

  acquire(player, now = Date.now()) {
    this._expire(now);
    if (now - this.minute.start >= 60000) this.minute = { start: now, count: 0 };
    let mine = 0;
    for (const lease of this.leases.values()) if (lease.player === player) mine += 1;
    if (mine >= this.perPlayer) return { ok: false, retry_after: 2 };
    if (this.leases.size >= this.maxInFlight) return { ok: false, retry_after: 3 };
    if (this.minute.count >= this.perMinute) {
      return { ok: false, retry_after: Math.max(1, Math.ceil((this.minute.start + 60000 - now) / 1000)) };
    }
    const id = `${now}-${this.next++}`;
    this.leases.set(id, { player, at: now });
    this.minute.count += 1;
    return { ok: true, id, mine: mine + 1 };
  }

  release(id) {
    this.leases.delete(id);
  }
}

// One instance for the whole relay (`idFromName("global")`), so every call
// passes the same count.
export class Gate {
  constructor(state, env) {
    this.limits = new Limits({
      maxInFlight: Number(env.MAX_IN_FLIGHT || 300),
      perMinute: Number(env.MAX_PER_MINUTE || 15000),
      perPlayer: Number(env.MAX_PER_PLAYER || 16),
    });
  }

  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === "/acquire") {
      return Response.json(this.limits.acquire(url.searchParams.get("player") || ""));
    }
    if (url.pathname === "/release") {
      this.limits.release(url.searchParams.get("id") || "");
      return Response.json({ ok: true });
    }
    return new Response("not found", { status: 404 });
  }
}
