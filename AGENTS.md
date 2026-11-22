# AGENTS.md

Guidance for coding agents working on uniqpick. Keep it current when you learn
something that would have saved you time.

## What this is

A command line tool that picks the most varied N images out of a folder, for
building style-LoRA training sets. Python 3.11, numpy and Pillow, plus CPU-only
torch and open_clip for the SigLIP2 embedder. No GUI, no server, no build step.

Entry point: `uniqpick/__main__.py` calls `uniqpick.cli.main`. `run.sh` is the
normal way to invoke it: it exports the repo root on `PYTHONPATH`, points
`HF_HOME` at the local cache when present, and runs `.venv/bin/python -m
uniqpick`.

## Setup and checks

- `./install.sh` creates `.venv` with `uv` (Python 3.11) and installs CPU-only
  wheels from `requirements.txt`. Do not add a dependency without adding it
  there; the axis backend needs nothing extra.
- Tests are plain `unittest`, there is no pytest config:
  `PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v`.
- The SigLIP2 model (`hf-hub:timm/ViT-B-16-SigLIP2`) is downloaded on first
  use. `run.sh` keeps `HF_HOME` if you set it and enables `HF_HUB_OFFLINE=1`
  when a cache is already present. There is no GPU requirement; everything
  runs on CPU.

## Layout

| path | what it is |
|---|---|
| `uniqpick/captions.py` | booru caption parsing: tags, weights, escapes, meta tags |
| `uniqpick/features.py` | md5, dhash, image size, RGB loading, SigLIP and basic embedders |
| `uniqpick/data.py` | discovery, the `Record` type, md5 plus dhash dedupe |
| `uniqpick/scoring.py` | TF-IDF tags, centred embeddings, `scale01`, aspect distance, `combine` |
| `uniqpick/select.py` | coverage / kcenter / disparity, greedy walks and the swap pass |
| `uniqpick/runstore.py` | `run/` format: `meta.json` plus `distances.npz` |
| `uniqpick/report.py` | `report.html`, `report.md`, `selection.csv`, `selection.json`, sheets |
| `uniqpick/cli.py` | argparse, `build_run` (compute), `finish` (pick), one-shot |
| `tests/test_pipeline.py` | end to end and unit tests, unittest style |

## Data flow

`compute` (`build_run`) does the expensive, k-independent half: discover,
dedupe, parse captions, embed images, build the three pairwise matrices, and
write `run/meta.json` plus `run/distances.npz`. `pick` (`finish`) loads the run,
rebuilds only the combined matrix, runs the selection, and renders the report.
The one-shot form does both and leaves the same run files behind.

## Invariants

Break one of these and the report silently lies.

- Every distance matrix is pairwise, symmetric, 0..1, zero diagonal. `load_run`
  checks shape, symmetry and the diagonal; keep it that way.
- `combine` sums the weighted signals raw and divides by the total applied
  weight once, at the very end. Weights are returned as configured (`tag 1.0`,
  `image 1.0`, `aspect 0.3`), not as shares of a normalized sum. Do not
  normalize tag and image against each other in the middle: the 1.0/1.0 balance
  and the soft aspect addition are a user decision.
- A uniform rescale of the combined distance cannot change the pick order, so
  the final division is only about keeping reported numbers in 0..1.
- A missing signal drops out and reports weight 0.0 for that side.
- Embeddings are centred by subtracting the corpus mean; raw cosine in a style
  set is 0.83-0.96 and cannot discriminate.
- The embedding cache key includes the file content md5, not mtime/size.
- Dedupe keeps the highest area copy and the earliest index; degenerate dhash
  values (all zeros or all ones) are never treated as near duplicates.
- `runstore.RUN_VERSION` must be bumped whenever the stored shape changes.
- In the report, only the tags+image view starts open; the single-signal views
  start collapsed. Anchors cannot nest: the card link wraps only the image, and
  the won/lost overlays are sibling links.

## Gotchas

- Run folders store absolute image paths. Moving the input folder or the run
  folder requires a fresh `compute`.
- `-k` larger than the corpus is clamped; `requested` and `count` in the report
  explain the gap.
- `--embeddings none` leaves no image distance, so the report has no image-only
  view; tests should not assume all three views exist.
- `aspect_distance` is `|log2(r1) - log2(r2)|` clipped at 1.0, so a doubling of
  the aspect ratio counts as one full step.
- The one-shot default output is `<parent>/<name>-uniqpick`; `pick` defaults to
  `<run>/k<N>` with a `-kcenter` or `-disparity` suffix for other methods.
- Contact sheets are 6 columns at 300 px; the thumbnails in `thumbs/` are what
  the report loads.

## Verifying a report change

Unit tests cover the markup, but the report is a webpage: after changing it,
regenerate one report from an existing run and check it in a browser. Good
cheap checks: the number of `<details class="sec">` blocks and which ones carry
`open`, that every unpicked image appears in exactly one won-against group, and
that a hover overlay contains a thumbnail link to the winning pick.

## Style

- English comments and docs, plain hyphens, no em dashes.
- Match the surrounding code: dataclasses, type hints, numpy in scoring and
  selection, string templates in the report.
- Keep diffs small. Do not bump versions or reformat unrelated code.
