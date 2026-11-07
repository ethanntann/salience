# Changelog

## 0.3.0 (2026-11-07): beta

- **Beta relay**: testers describe with the project's key through a Cloudflare
  Worker, from an invite code; 5 hours of matches a month each.
- **One-click installer**: Python, the app and FFmpeg in `Salience-Setup.exe`,
  per user, no admin rights.
- **Clips you like**: import clips or montages to start a new player's taste.
- **OBS recordings**: highlights from the player's own OBS recordings.
- **Export my stats**: counts a tester can send back, nothing else.
- **Storage**: recording stops under 10 GB free; lobby footage is deleted
  during play; a session's raw footage goes 3 days after it is fully rated,
  or after 7 days.
- **Security**: the relay forwards only Salience's own requests and caps each
  player's spend; the app's local server answers only its own window; the
  installer's Python download and packages are pinned and checked.

## 0.2.0

- Highlights from every 8 seconds of a match, ranked by the player's own
  ratings; the desktop app and tray recorder.
