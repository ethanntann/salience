"""Pipeline-neutral media utilities.

These are the only pieces of the original local student/teacher codebase that
both pipelines needed: FFmpeg frame extraction, FFprobe metadata, and the
weapon-name ontology. They live here so that `backend/` can be removed without
taking v2 down with it. Nothing in this package imports either pipeline.
"""
