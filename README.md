# Salience

**[Open the live demo](https://salience-demo.onrender.com)** | **[View the source](https://github.com/ethanntann/salience)**

> Render's free instance may sleep after inactivity. Allow roughly 50 seconds
> for the first request to wake it up.

Salience is a local-first AI highlight ranking layer for Fortnite clips. It
watches the folder where tools like ShadowPlay, SteelSeries Moments, OBS, or
Medal already save clips, analyzes each clip, ranks the moments worth
reviewing, and learns from user feedback.

Auto-clippers save events. Salience learns whether you actually care about
them.

## Teacher / student architecture — what's cloud and what's local

Salience uses two models, for two different jobs:

- **Teacher (development-time, cloud):** [Fireworks AI](https://fireworks.ai)
  serving Qwen VLM. During development, sampled keyframes from labeled clips
  are sent to the teacher, which returns structured labels — weapon,
  elimination type, victory, context, event timestamps. This is how the
  training data was built. It is **not** required to run or judge the app.
- **Student (shipped, local, CPU-only):** a ~9 MB MobileNetV3-small backbone
  with an event-locator head, weapon/evidence heads, and context heads,
  distilled from the teacher's labels and exported to ONNX
  (`student-artifacts/`). It runs entirely on CPU with `onnxruntime`, fused
  with local OCR (`RapidOCR`) for the weapon HUD and victory banner. **No
  API key, no cloud call, no upload of your gameplay.**

On an 81-clip held-out eval, the student agrees with the cloud teacher on
~80% of labels on average (enemy-visible F1 0.97, elimination F1 0.93,
victory F1 0.89). The eval harness and latest report are in
`.local-data/student/` after you run it yourself, or see
`submission/agreement-report-student-v8.json` for the frozen snapshot used in
this submission.

**The shipped product only needs the student.** The teacher is an optional,
swappable development tool — `SALIENCE_VLM_PROVIDER` can point at Fireworks,
a local endpoint, or AMD Developer Cloud (Qwen2.5-VL via vLLM), but the
default judge-facing configuration (`docker-compose.yml`) uses
`SALIENCE_VLM_PROVIDER=local` and never calls out to the internet.

## AMD, Fireworks, and external-service usage

| Resource | How Salience uses it | Required for the submitted demo? |
| --- | --- | --- |
| **Fireworks AI** | Served the Qwen vision-language teacher during development. Sampled gameplay keyframes were sent to it to produce structured weapon, event, timestamp, and context labels used to train and evaluate the local student. It remains available for optional future labeling. | **No.** It is disabled in Docker and on Render. |
| **AMD Developer Cloud** | The code includes an OpenAI-compatible Qwen2.5-VL/vLLM teacher client (`AmdDeveloperCloudTeacherClient`) and hardware reporting for AMD/Radeon devices. This is an alternative development-time teacher and a path toward future AMD acceleration. | **No.** It was not active for the frozen evaluation or submitted Render runtime. |
| **Render** | Hosts the public Docker demo from `render.yaml`, runs the local student, and exposes `/health`. | **Yes, only for the hosted URL.** Local Docker use does not require Render. |
| **GitHub** | Hosts the source, small student artifacts, and judge sample clips; pushes to `main` trigger the Render deployment. | **Only to obtain the source.** |
| **FFmpeg, OpenCV, ONNX Runtime, RapidOCR** | Decode video, sample frames, execute the student model, and read HUD/victory text. They are installed inside the Docker image. | **Yes, but Docker installs them automatically.** |

No Fireworks or AMD credential is stored in this repository or configured in
the public demo. The submitted runtime performs inference locally inside its
Render container; gameplay is not forwarded to either teacher service.

## Quick start

Prerequisite: Docker Desktop or Docker Engine with Compose.

```bash
docker compose up --build
```

Then open <http://localhost:7860>.

Optional health check:

```bash
curl http://localhost:7860/health
# {"status":"ok"}
```

This uses `docker-compose.yml` (tracked in git, safe on any machine):

- `SALIENCE_DEMO_MODE=true` — seeds a small synthetic ranked inbox instantly
  so there's something to review immediately.
- `SALIENCE_VLM_PROVIDER=local` — any clip you scan gets labeled by the local
  ONNX student, not a cloud API. No key needed.
- The trained student model (`student-artifacts/`) and 10 unseen sample
  clips (`sample-clips/`) are baked into the image, so the container is
  fully self-contained.

## Trying the local student model on new clips

`sample-clips/` ships with 10 real gameplay clips the student has never seen
in training or evaluation (see `sample-clips/README.md`). With the stack
running, open <http://localhost:7860>, find the **Process new clips** panel,
pick "Sample clips (judges)" from the dropdown, and click **Process clips**.
The ranked inbox updates with labels and scores produced entirely by the
local student model — no API key, no cloud call.

The same panel can target `demo-video/`, a local-only folder used for rehearsal
footage. Its MP4 files are intentionally excluded from git; add your own files
there before building if you want to use that option.

Equivalent API call, if you'd rather script it:

```bash
curl -X POST http://localhost:7860/folders/scan \
  -H "Content-Type: application/json" \
  -d '{"path": "/app/sample-clips", "enrich": true}'
```

To add your own clips instead, drop `.mp4` files into `sample-clips/` before
`docker compose build` and rebuild.

## Using the real Fireworks teacher (optional, not required)

```bash
# in a .env file at the repo root:
SALIENCE_VLM_PROVIDER=fireworks
FIREWORKS_API_KEY=your_key
FIREWORKS_MODEL=accounts/fireworks/models/qwen3p7-plus
```

`docker-compose.yml` reads `FIREWORKS_API_KEY`/`FIREWORKS_MODEL` from the
environment (default provider is `local`, so this is opt-in). Never commit a
real key — `.env` is gitignored.

### Using AMD Developer Cloud instead (optional, not required)

```bash
# in a .env file at the repo root:
SALIENCE_VLM_PROVIDER=amd-developer-cloud
AMD_DEVELOPER_CLOUD_API_KEY=your_key
AMD_DEVELOPER_CLOUD_BASE_URL=https://your-amd-endpoint.example/v1
AMD_DEVELOPER_CLOUD_MODEL=Qwen/Qwen2.5-VL-72B-Instruct
```

This selects the AMD-hosted teacher for development labeling. It is not used
by the public demo or the default local setup.

## Main code paths

- `backend/salience_api/app.py` - FastAPI entry point and end-to-end request flow.
- `backend/salience_api/student/local_teacher.py` - local student inference client.
- `backend/salience_api/student/onnx_runtime.py` - ONNX model execution.
- `backend/salience_api/clips/indexer.py` - clip discovery and indexing.
- `backend/salience_api/ranking/highlights.py` - highlight hierarchy construction.
- `backend/salience_api/ranking/scoring.py` and `personal_ranker.py` - base and personalized ranking.
- `frontend/src/App.tsx` - React application entry point.
- `frontend/src/components/ReviewInbox.tsx` - primary ranked review UI.
- `Dockerfile`, `docker-compose.yml`, and `render.yaml` - container and deployment definitions.

## Architecture

```text
Clip folder or demo data
  -> clip registry
  -> feature extraction (OpenCV/FFmpeg keyframes, RapidOCR)
  -> teacher labels (Fireworks VLM, dev-time) OR local ONNX student (shipped)
  -> local SQLite
  -> base scorer
  -> personal ranker (learns from your feedback)
  -> ranked review UI
  -> feedback loop
```

## Local Development

Backend:

```bash
cd backend
python -m pip install -e ".[dev]"
uvicorn salience_api.app:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server calls the backend at `http://localhost:8000`.

## Privacy

The shipped product runs locally by default. Your clips do not need to leave
your PC for feature extraction, ranking, or personalization — the default
judge-facing configuration never calls a cloud API. The Fireworks/AMD teacher
integrations are optional, development-time tools you can enable explicitly
with your own key.

## Roadmap

- **Next:** upload any file or montage -> ranked highlight breakdown (no
  clips folder needed)
- **Soon:** Valorant support — the teacher-to-student pipeline is
  game-agnostic; each new game is a labeling pass and a small student head,
  not a new product
- **Later:** Windows tray app (watch folder, one-click), AMD/DirectML
  acceleration, and continuous teacher-label refresh so the local model keeps
  improving without ever uploading user gameplay by default
