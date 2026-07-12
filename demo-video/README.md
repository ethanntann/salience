# demo-video

Private rehearsal folder. Drop test clips here while practicing the "process
new clips" moment before recording — separate from `sample-clips/`, which is
the frozen 10-clip set shipped for judges.

Video files in this folder are gitignored (only this note is tracked), so
rehearsal footage never gets committed or pushed. The "Process clips" dropdown
in the app can target this folder (`/app/demo-video` in the container) or
`sample-clips/` (`/app/sample-clips`) — pick whichever you're testing.

Rebuild the image after changing the contents (`docker compose build`) since
clips are baked in at build time, same as `sample-clips/`.
