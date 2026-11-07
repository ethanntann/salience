// Who may use the relay and what they have used, in D1 (`schema.sql`).
// `MemoryStore` is the same thing in memory, for the tests.

export class D1Store {
  constructor(db) {
    this.db = db;
  }

  findInstall(hash) {
    return this.db.prepare("select tier, disabled from installs where token_hash = ?").bind(hash).first();
  }

  async redeemInvite(code) {
    const result = await this.db
      .prepare("update invites set uses_left = uses_left - 1 where code = ? and uses_left > 0")
      .bind(code).run();
    return (result.meta && result.meta.changes) > 0;
  }

  addInstall(hash, invite, now) {
    return this.db.prepare("insert into installs (token_hash, invite, tier, created_at, disabled) values (?, ?, 'free', ?, 0)")
      .bind(hash, invite, now).run();
  }

  async monthUsage(hash, month) {
    const row = await this.db.prepare("select calls, cost_usd from usage where token_hash = ? and month = ?")
      .bind(hash, month).first();
    return { calls: row ? row.calls : 0, cost_usd: row ? row.cost_usd : 0 };
  }

  async monthTotal(month) {
    const row = await this.db.prepare("select cost_usd from totals where month = ?").bind(month).first();
    return row ? row.cost_usd : 0;
  }

  charge(hash, month, cost) {
    return this.db.batch([
      this.db.prepare(`insert into usage (token_hash, month, calls, cost_usd) values (?, ?, 1, ?)
        on conflict (token_hash, month) do update set calls = calls + 1, cost_usd = cost_usd + excluded.cost_usd`)
        .bind(hash, month, cost),
      this.db.prepare(`insert into totals (month, calls, cost_usd) values (?, 1, ?)
        on conflict (month) do update set calls = calls + 1, cost_usd = cost_usd + excluded.cost_usd`)
        .bind(month, cost),
    ]);
  }

  addInvite(code, uses, note) {
    return this.db.prepare("insert into invites (code, uses_left, note) values (?, ?, ?)").bind(code, uses, note).run();
  }

  setDisabled(prefix, disabled) {
    return this.db.prepare("update installs set disabled = ? where token_hash like ?").bind(disabled ? 1 : 0, `${prefix}%`).run();
  }

  async report(month) {
    const players = await this.db.prepare(`select substr(i.token_hash, 1, 12) as id, i.invite, i.tier, i.disabled,
        coalesce(u.calls, 0) as calls, coalesce(u.cost_usd, 0) as cost_usd
      from installs i left join usage u on u.token_hash = i.token_hash and u.month = ? order by calls desc`)
      .bind(month).all();
    return { month, total_usd: await this.monthTotal(month), players: players.results };
  }
}

export class MemoryStore {
  constructor() {
    Object.assign(this, { installs: new Map(), invites: new Map(), usageRows: new Map(), totals: new Map() });
  }

  async findInstall(hash) {
    return this.installs.get(hash) || null;
  }

  async redeemInvite(code) {
    const invite = this.invites.get(code);
    if (!invite || invite.uses_left <= 0) return false;
    invite.uses_left -= 1;
    return true;
  }

  async addInstall(hash, invite) {
    this.installs.set(hash, { tier: "free", disabled: 0, invite });
  }

  async monthUsage(hash, month) {
    return this.usageRows.get(`${hash}|${month}`) || { calls: 0, cost_usd: 0 };
  }

  async monthTotal(month) {
    return (this.totals.get(month) || { cost_usd: 0 }).cost_usd;
  }

  async charge(hash, month, cost) {
    const row = await this.monthUsage(hash, month);
    this.usageRows.set(`${hash}|${month}`, { calls: row.calls + 1, cost_usd: row.cost_usd + cost });
    const total = this.totals.get(month) || { calls: 0, cost_usd: 0 };
    this.totals.set(month, { calls: total.calls + 1, cost_usd: total.cost_usd + cost });
  }

  async addInvite(code, uses, note) {
    this.invites.set(code, { uses_left: uses, note });
  }

  async setDisabled(prefix, disabled) {
    for (const [hash, install] of this.installs) if (hash.startsWith(prefix)) install.disabled = disabled ? 1 : 0;
  }

  async report(month) {
    return { month, total_usd: await this.monthTotal(month), players: [...this.installs.keys()] };
  }
}
