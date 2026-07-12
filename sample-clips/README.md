# sample-clips

Ten unseen Fortnite clips, shipped with the repo so judges can run the local
student model themselves without needing their own gameplay footage. These
clips were never used in training or evaluation — they exist to prove the
local pipeline (keyframe extraction -> ONNX student -> OCR -> ranked inbox)
works on genuinely new footage, not just the eval split.

They are baked into the Docker image at `/app/sample-clips` (see
`Dockerfile`). With the stack running (`docker compose up --build`, then
open http://localhost:7860), process them through the LOCAL student model —
no API key, no cloud calls — with:

```bash
curl -X POST http://localhost:7860/folders/scan \
  -H "Content-Type: application/json" \
  -d '{"path": "/app/sample-clips", "enrich": true}'
```

Then reload http://localhost:7860 to see the new clips ranked in the inbox
with student-generated labels and scores.

> Adding your own clips: drop `.mp4` files in this folder before
> `docker compose build` (or `docker build`) so they get copied into the
> image. Keep each file well under GitHub's 100 MB per-file limit — trim to
> the highlight moment and re-encode at a lower bitrate if a raw recording is
> larger.
