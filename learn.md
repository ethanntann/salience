# Learning the current project

Start with [the v2 guide](pipeline_v2/README.md) and [scope](pipeline_v2/SCOPE.md).

1. `cli._prepare` → `frames.window_frames` → `vlm.describe`: model inputs.
2. `store.upsert_clip` → `embed.load_encoder`: persistence and representation.
3. `review._next_clip` → `ranking.ranked_fights`: decisions and inbox behavior.
4. `taste.pairwise_accuracy`: what is held out and what the metric tests.
5. `capture.daemon` → `capture.trigger` → `capture.recorder`: separate experiment.

Run `python -m pytest` before changing persistence/capture lifecycle. Rank scores
are not calibrated probabilities. Description accuracy, ranking quality, live
recall, and recorder performance are different measurements.
The [v1 learning guide](archive/v1-learning-guide.md) is historical reference.
