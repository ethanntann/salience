-- The relay's records. Tokens are kept only as SHA-256 hashes.
create table if not exists installs (
  token_hash text primary key,
  invite     text,
  tier       text not null default 'free',
  created_at text not null,
  disabled   integer not null default 0
);
create table if not exists invites (
  code      text primary key,
  uses_left integer not null,
  note      text
);
-- Calls per player per month: each is 8 seconds of their allowance.
create table if not exists usage (
  token_hash text not null,
  month      text not null,
  calls      integer not null,
  cost_usd   real not null,
  primary key (token_hash, month)
);
-- Everyone's spend this month, against the relay's own cap.
create table if not exists totals (
  month    text primary key,
  calls    integer not null,
  cost_usd real not null
);
