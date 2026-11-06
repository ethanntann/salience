"""Local-only review for a frozen evaluation; labels never enter training."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

from pipeline_v2.jsonio import read_json, utc_now, write_json
from pipeline_v2.http_media import serve_video


def validate_labels(payload: dict, ids: set[str], duration: float) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Expected object")
    labels = payload.get("labels", {})
    events = payload.get("events", [])
    if (
        not isinstance(labels, dict)
        or not isinstance(events, list)
        or len(events) > 100
    ):
        raise ValueError("Invalid labels/events")
    for key, label in labels.items():
        if key not in ids or not isinstance(label, dict):
            raise ValueError("Unknown window")
        if label.get("keep") not in ("keep", "skip", "unsure"):
            raise ValueError("Invalid preference")
        if label.get("accuracy", "unreviewed") not in (
            "unreviewed",
            "accurate",
            "minor_error",
            "major_error",
            "unsure",
        ):
            raise ValueError("Invalid accuracy")
        if (
            not isinstance(label.get("notes", ""), str)
            or len(label.get("notes", "")) > 2000
        ):
            raise ValueError("Invalid notes")
    for e in events:
        if (
            not isinstance(e, dict)
            or not isinstance(e.get("start"), (int, float))
            or not isinstance(e.get("end"), (int, float))
        ):
            raise ValueError("Invalid event")
        if not 0 <= e["start"] < e["end"] <= duration:
            raise ValueError("Invalid event interval")
    return {
        "annotator": "user",
        "saved_utc": utc_now(),
        "labels": labels,
        "events": events,
        "timeline_review_complete": payload.get("timeline_review_complete") is True,
    }


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = Path(self.translate_path(self.path)).resolve()
        if path.suffix.lower() == ".mp4":
            if (
                not path.is_relative_to(Path(self.directory).resolve())
                or not path.is_file()
            ):
                self.send_error(404)
                return
            serve_video(self, path)
            return
        super().do_GET()

    def do_POST(self):
        expected = f"http://{self.headers.get('Host')}"
        if self.path != "/labels" or self.headers.get("Origin") != expected:
            self.send_error(403)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 200_000:
                raise ValueError("Invalid request size")
            out = Path(self.directory)
            data = read_json(out / "review-data.json")
            payload = validate_labels(
                json.loads(self.rfile.read(size)),
                {w["id"] for w in data["windows"]},
                data["duration"],
            )
            write_json(out / "user-labels.json", payload)
            if (out / "metrics.json").exists():
                from pipeline_v2.evaluation_metrics import report

                report(out)
        except (ValueError, TypeError, OSError) as exc:
            self.send_error(400, str(exc))
            return
        self.send_response(204)
        self.end_headers()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def build(out: Path) -> None:
    plan = read_json(out / "plan.json")
    scores = {r["id"]: r["score"] for r in read_json(out / "ranking.json")["scores"]}
    windows = read_json(out / "prepared.json")["windows"]
    for w in windows:
        w["description"] = read_json(out / "results" / f"{w['id']}.json")["description"]
        w["score"] = scores[w["id"]]
        w["preview"] = w["preview"].replace("\\", "/")
        for f in w["images"]:
            f["path"] = f["path"].replace("\\", "/")
    write_json(
        out / "review-data.json",
        {
            "session_id": plan["session_id"],
            "windows": windows,
            "duration": max(w.get("timeline_end", 0) for w in windows),
            "review_note": plan.get("review_note", ""),
            "cohort_labels": plan.get("cohort_labels", {}),
            "has_timeline": (out / "continuous.mp4").is_file(),
        },
    )
    if not (out / "user-labels.json").exists():
        write_json(
            out / "user-labels.json", {"annotator": "user", "labels": {}, "events": []}
        )
    (out / "index.html").write_text(HTML, encoding="utf-8")


HTML = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Salience · Unseen footage review</title><style>
:root{color-scheme:dark;font:16px system-ui;background:#10151d;color:#e7edf5}[hidden]{display:none!important}body{max-width:1250px;margin:auto;padding:24px}h1{font-size:28px;margin-bottom:8px}p{line-height:1.5;color:#b8c7d9}button,select,textarea{font:inherit;background:#202e40;color:#eef4ff;border:1px solid #43536a;border-radius:7px;padding:9px;margin:4px}button{cursor:pointer}button:hover{background:#31465f}button:disabled{opacity:.5;cursor:default}.layout{display:grid;grid-template-columns:2fr 1fr;gap:22px}video{width:100%;background:black;max-height:560px}textarea{box-sizing:border-box;width:100%;min-height:80px}article{padding:18px;background:#182230;border-radius:10px}.frames{display:flex;overflow:auto;gap:8px}.frames img{width:210px}#model{white-space:pre-wrap;line-height:1.6}#saved{color:#91d6b4}#events{font-size:14px}#warning{background:#34242a;padding:10px;border-radius:6px}label{display:block}a{color:#9fc7ff}@media(max-width:850px){.layout{grid-template-columns:1fr}}
</style><h1>Unseen footage review</h1><p>Watch first. Choose what you would keep, then reveal v2's description. Your judgments save locally and stay outside training.</p>
<p id="warning">This recorder caused FPS drops and crashes. It is off. Reviewing saved footage does not start capture.</p>
<p id="studyNote"></p><button id="windowMode">Review windows</button><button id="timelineMode" hidden>Watch continuous sequence</button><span id="saved"></span>
<section id="timeline" hidden><h2>Mark moments you would want saved</h2><p>Watch the whole sequence, including downtime. Mark each wanted moment before looking at rankings. Timing below follows the continuous three-minute stretch.</p>
<video id="whole" controls preload="metadata"></video><button id="markStart">Start moment here</button><button id="markEnd">End and save moment</button><span id="marker"></span><div id="events"></div><label><input type="checkbox" id="timelineComplete"> I watched the whole sequence and marked every moment I would keep (including none).</label></section>
<section id="windowReview"><p><select id="cohort" aria-label="Review group"></select><button id="prev">Previous</button><button id="next">Next</button><span id="position"></span></p>
<div class="layout"><div><video id="clip" controls preload="metadata"></video><button id="replay">Replay model window</button><button id="context" hidden>Play with context</button><p id="bounds"></p></div><article>
<h2>Your judgment</h2><label>Would you keep this moment?<select id="keep"><option value="">Choose…</option><option value="keep">Keep</option><option value="skip">Skip</option><option value="unsure">Unsure</option></select></label>
<textarea id="notes" placeholder="What happened? Any detail that matters to you?" aria-label="Review notes"></textarea><button id="save">Save judgment</button><button id="reveal">Reveal v2</button><div id="model" hidden></div>
<label id="accuracyWrap" hidden>Description accuracy<select id="accuracy"><option value="unreviewed">Choose…</option><option value="accurate">Accurate</option><option value="minor_error">Minor error</option><option value="major_error">Major error</option><option value="unsure">Unsure</option></select></label></article></div>
<p>The seven images v2 saw. Click one to inspect it at full size.</p><div class="frames" id="frames"></div></section>
<p id="progress"></p><p>Scores use the v2 TF-IDF baseline trained on historical preferences. They are not calibrated confidence or a reproduction of the historical semantic-encoder result.</p>
<script>
let data,review,items,index=0,mark=null,playbackEnd=0,saveChain=Promise.resolve();const $=id=>document.getElementById(id);const video=$('clip');
function current(){return items[index]}
function message(s){$('saved').textContent=s}
function persist(){const body=JSON.stringify(review);message('Saving…');saveChain=saveChain.catch(()=>{}).then(async()=>{const r=await fetch('/labels',{method:'POST',headers:{'Content-Type':'application/json'},body});if(!r.ok)throw Error(await r.text());message('Saved locally');});saveChain.catch(e=>message('Save failed: '+e.message));return saveChain}
function progress(){const count=data.windows.filter(w=>review.labels[w.id]).length;$('progress').textContent=count+' / '+data.windows.length+' windows judged'+(data.has_timeline?' · '+review.events.length+' wanted moments marked':'')}
function render(){video.pause();const w=current(),r=review.labels[w.id]||{};playbackEnd=w.playback_end??w.end;video.onloadedmetadata=()=>{video.currentTime=w.playback_start??w.start};video.src=w.preview;$('context').hidden=w.playback_start===undefined;$('position').textContent=(index+1)+' / '+items.length+' · '+w.id;$('bounds').textContent='Model window: '+w.start.toFixed(2)+'–'+w.end.toFixed(2)+'s in source chunk.'+(w.playback_start!==undefined?' Context playback: '+w.playback_start.toFixed(2)+'–'+w.playback_end.toFixed(2)+'s. Rate the description against the model window and its seven frames.':' Playback pauses at the window end.');$('keep').value=r.keep||'';$('notes').value=r.notes||'';$('accuracy').value=r.accuracy||'unreviewed';$('model').hidden=true;$('accuracyWrap').hidden=true;$('model').textContent='';$('frames').replaceChildren();for(const f of w.images){const a=document.createElement('a');a.href=f.path;a.target='_blank';const im=document.createElement('img');im.src=f.path;im.alt='Frame at '+f.timestamp.toFixed(2)+'s';a.append(im);$('frames').append(a)}$('prev').disabled=index===0;$('next').disabled=index===items.length-1;progress()}
async function save(){if(!$('keep').value){message('Choose Keep, Skip or Unsure first.');return false}const old=review.labels[current().id]||{};review.labels[current().id]={...old,keep:$('keep').value,notes:$('notes').value,accuracy:$('accuracy').value};await persist();progress();return true}
function renderEvents(){$('events').replaceChildren();review.events.forEach((e,i)=>{const row=document.createElement('p');row.textContent=e.start.toFixed(1)+'–'+e.end.toFixed(1)+'s ';const b=document.createElement('button');b.textContent='Remove';b.onclick=async()=>{review.events.splice(i,1);await persist();renderEvents();progress()};row.append(b);$('events').append(row)})}
$('cohort').onchange=()=>{items=data.windows.filter(w=>w.cohort===$('cohort').value);index=0;render()};$('prev').onclick=()=>{index--;render()};$('next').onclick=()=>{index++;render()};$('replay').onclick=()=>{playbackEnd=current().end;video.currentTime=current().start;video.play()};$('context').onclick=()=>{playbackEnd=current().playback_end;video.currentTime=current().playback_start;video.play()};video.ontimeupdate=()=>{if(video.currentTime>=playbackEnd&&!video.paused)video.pause()};$('save').onclick=()=>save().catch(e=>message(e.message));$('accuracy').onchange=()=>save().catch(e=>message(e.message));
$('reveal').onclick=async()=>{try{if(!await save())return;review.labels[current().id].revealed=true;await persist();const w=current(),d=w.description;$('model').textContent=d.summary+'\n\nOutcome: '+d.outcome+' · Model confidence: '+d.confidence+'\nRanking score: '+w.score.toFixed(3);$('model').hidden=false;$('accuracyWrap').hidden=false}catch(e){message(e.message)}};
$('timelineMode').onclick=()=>{video.pause();$('timeline').hidden=false;$('windowReview').hidden=true};$('windowMode').onclick=()=>{$('whole').pause();$('timeline').hidden=true;$('windowReview').hidden=false};$('markStart').onclick=()=>{mark=$('whole').currentTime;$('marker').textContent='Start: '+mark.toFixed(1)+'s'};$('markEnd').onclick=async()=>{const end=$('whole').currentTime;if(mark===null||end<=mark){message('Mark a start before the end.');return}review.events.push({start:mark,end});mark=null;$('marker').textContent='';await persist();renderEvents();progress()};
$('timelineComplete').onchange=async()=>{review.timeline_review_complete=$('timelineComplete').checked;await persist()};
Promise.all([fetch('review-data.json').then(r=>r.json()),fetch('user-labels.json').then(r=>r.json())]).then(([d,r])=>{data=d;review=r;$('studyNote').textContent=d.review_note||'';$('timelineMode').hidden=!d.has_timeline;if(d.has_timeline)$('whole').src='continuous.mp4';$('timelineComplete').checked=!!r.timeline_review_complete;for(const cohort of new Set(d.windows.map(w=>w.cohort))){const option=document.createElement('option');option.value=cohort;option.textContent=(d.cohort_labels?.[cohort]||cohort)+' ('+d.windows.filter(w=>w.cohort===cohort).length+')';$('cohort').append(option)}items=data.windows.filter(w=>w.cohort===$('cohort').value);render();renderEvents()}).catch(e=>message(e.message));
</script></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--port", type=int, default=8778)
    args = parser.parse_args()
    out = args.out.resolve()
    build(out)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), partial(Handler, directory=str(out))
    )
    print(f"Review: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
