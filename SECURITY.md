# Security

## Reporting a problem

Please report security problems privately, through GitHub's
[private vulnerability reporting](../../security/advisories/new) for this
repository, not in a public issue. You'll get a reply within a few days.

## What Salience protects

- **The project's Fireworks key** lives only as a secret of the beta relay
  (`relay/`, a Cloudflare Worker). It is never in this repository, the app or
  the installer.
- **Each install's relay token** is random (256 bits), stored on the relay only
  as a SHA-256 hash, and limited to that player's monthly allowance: hours of
  matches and a dollar cap. The relay forwards only Salience's own request
  shape (its model, its fields, text and inline frames, nothing fetched), and
  stops itself at a monthly budget.
- **The desktop app's local server** listens on 127.0.0.1 only, checks the
  `Host` header (against DNS rebinding) and requires a secret made at each
  launch, which only the app's own window has.
- **Your recordings** stay on your PC. Only gameplay frames are sent to the
  vision model; audio, transcripts, inputs and ratings never leave it.

## Supported versions

Only the latest release is supported during the beta.
