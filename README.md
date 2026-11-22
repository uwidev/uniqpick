# uniqpick

Pick the most *varied* N images out of a folder, for building a style LoRA
training set where one feature (one character, one palette, one composition)
does not dominate.

Two questions drive every pick:

1. **What is different from everything already chosen?** The
   largest-distance / k-sampling idea.
2. **Is this image unusual in the corpus?** Uniqueness from caption tags and
   from image features.

The tool combines both signals, selects greedily, improves the set with a swap
pass, and explains every decision in a browsable report: which images were
picked, which tags the picks miss, and what each rejected image lost to.

## Requirements

- Python 3.11
- numpy and Pillow (the basic backend needs only these)
- torch, torchvision, open_clip and timm for the default SigLIP2 backend
  (CPU wheels are enough; no GPU required)
- [uv](https://docs.astral.sh/uv/) for `install.sh`, or install the packages in
  `requirements.txt` yourself

The first SigLIP2 run downloads about 1.5 GB of model weights from Hugging
Face. The cache is reused afterwards and the tool runs fully offline.

## Install

```bash
git clone <your-repo-url> uniqpick
cd uniqpick
./install.sh          # uv venv .venv --python 3.11, then CPU-only torch + deps
./run.sh --help
```

`install.sh` takes a couple of minutes because it fetches CPU torch. The app is
self-contained: its own `.venv`, no dependency on any other project. Point
`UNIQPICK_PYTHON` at another interpreter to run outside the bundled venv.

## Quick start

```bash
./run.sh /path/to/images -k 24
```

Output goes to `../<folder>-uniqpick/` next to the input; override it with
`-o`. Captions are read from `<image>.txt` and `<image>.personal.txt` next to
each image. The default backend is SigLIP2 embeddings on CPU; if the model
cannot be loaded the tool falls back to cheap colour/layout features and says
so in the log and the report.

## Compute once, pick many

The expensive half (hashing, caption parsing, embeddings, distance matrices)
does not depend on `k`, so it is worth doing once:

```bash
./run.sh compute /path/to/images -o run/     # embeddings + distances, once
./run.sh pick run/ -k 24  -o run/k24         # seconds: selection + report
./run.sh pick run/ -k 64  -o run/k64
./run.sh pick run/ -k 96  --method disparity
./run.sh pick run/ -k 24  --aspect-weight 0  # reweight without recomputing
```

`compute` writes `run/distances.npz` (tag, image and frame-shape matrices) and
`run/meta.json` (records, captions, duplicates, vocabulary, default weights).
`pick` loads those and only runs the selection and the report. Recompute only
when the corpus changes, or when you change something that feeds a distance:
`--model`, `--embeddings`, `--dedupe-hamming`, `--caption-suffix`, `--min-df`
or `--max-df`. At pick time `--exclude-tag` is cheap because it rebuilds just
the tag matrix.

The one-shot form leaves the same run files in its output folder, so an old
output can be picked again. Run folders store absolute image paths, so moving
the input folder requires a fresh `compute`.

## What you get

```
<out>/
  selected/            copies (or symlinks with --link) plus matching captions
  selected_sheet.png   numbered contact sheet of the picks
  report.html          gallery: picks, tag coverage, duplicates, rejected cards
  report.md            same content as text
  selection.csv        one row per image, with scores and nearest-selected info
  selection.json       machine-readable selection
  thumbs/              small versions used by the report
```

`report.html` is the one to open. It shows each pick with its rank, uniqueness
and how far it is from everything else, then:

- tag coverage: tags frequent in the corpus but missing from the selection,
  the usual reason a LoRA under-learns part of the style
- an aspect and size table: frame-shape distribution of the corpus versus the
  picks, plus images that will be upscaled at training resolution
- the tags-only and image-only selections, as comparison views
- hover panels: each pick lists the images it won against, and each rejected
  image names the pick it lost to and by how much

The tags+image list opens by default. The tags-only and image-only lists start
collapsed; open one from its heading or with Expand all.

## How it scores

**Tag distance.** Captions are parsed into tags (`(tag:1.2)`, `[tag]`, `{tag}`,
`\_` escapes, `_` as space). Tags are TF-IDF weighted, so tags shared by almost
every image (the artist tag, the rating, `1girl`) carry no weight and cannot
discriminate. Meta tags (`high resolution`, `absurd resolution`, ...) are
dropped outright.

**Image distance.** Embeddings of the images, *centred by subtracting the
corpus mean* and then compared by cosine. In a style dataset every image is
close to every other one, so raw similarity is nearly constant; centring
removes the shared direction and leaves what actually differs. A dependency
light backend (colour histograms, layout grid, edge energy) is available with
`--embeddings basic` and needs only numpy and Pillow.

**Frame shape** is a third, softer signal. A DiT trains on aspect buckets: every
image is resampled before the model sees it, so pixel count stops mattering,
but the frame shape is learned along with the style. A corpus that is 90%
portrait teaches portrait composition as part of the look. `--aspect-weight`
(default 0.3) is added on top of the tag and image distances, which each span
0..1: framing can add up to 0.30 of distance at a full doubling. It is a soft
term, not a quota: buckets are not guaranteed a pick, the picks are only
nudged apart in shape.

**Combined distance** is the weighted sum of the signals, default tag 1.0,
image 1.0, aspect 0.3. Nothing is normalized in the middle, so tag and image
stay equal partners and aspect stays a soft addition. The final distance is
divided by that total weight once at the end, which keeps the reported numbers
in 0..1 next to the per-signal ones; a uniform rescale cannot change the pick
order.

**Selection.** Default `--method coverage` is greedy facility location: each
new pick is the image that most improves the representation of the whole
corpus. That keeps the set representative instead of collecting oddities.
`--method kcenter` (farthest point sampling) maximizes the smallest distance
between two picks: maximum spread, but it surfaces the most atypical images, a
poor default for training data. `--method disparity` maximizes the summed
pairwise distance (sum-of-pairs): it spreads picks more evenly than kcenter and
still leans to the edges of the corpus.

After the greedy walk every method runs a swap pass: a pick and an unpicked
image trade places while the objective rises, so the reported set is never
worse than the greedy one. `--no-refine` keeps the raw greedy order (the pick
sequence is then guaranteed to be a prefix: k=8 is the first half of k=16). The
report shows the method as `coverage + swap` and prints the closest pair among
picks, which is your redundancy check on the result.

**Uniqueness** is a separate per-image score: mean distance to all other kept
images. It is reported for every image, selected or not.

**Dedupe** collapses exact duplicates (md5) and near duplicates (dhash Hamming
distance, default 6 bits), keeping the highest resolution copy. Images whose
dhash is degenerate (flat colour or blank) are never merged this way.

## Options worth knowing

| flag | meaning |
|---|---|
| `-k, --count` | how many to pick (clamped to the corpus size) |
| `--method coverage\|kcenter\|disparity` | representative spread vs maximum spread vs sum-of-pairs |
| `--no-refine` | skip the swap pass, keep the raw greedy order |
| `--tag-weight 1.0`, `--image-weight 1.0` | balance text vs pixels |
| `--aspect-weight 0.3` | distance added on top of tag and image by frame shape; `0` disables |
| `--train-resolution 1024` | used by the report to flag images that will be upscaled |
| `--embeddings auto\|clip\|basic\|none` | image features; `none` = tags only |
| `--model` | any open_clip model id, default `hf-hub:timm/ViT-B-16-SigLIP2` |
| `--exclude-tag TAG` | drop a tag before scoring, repeatable (e.g. a rating) |
| `--dedupe-hamming N` | near-duplicate threshold; `-1` disables it |
| `--max-df 0.95` | ignore tags appearing in more than 95% of captions |
| `--recursive` | scan subfolders |
| `--link` | symlink into `selected/` instead of copying |
| `--no-cache` | recompute embeddings (cache lives in the run or output folder) |

## Tuning

- If the selection looks too samey, raise `--image-weight`, or re-pick with
  `--method disparity` and compare sheets. Check the `closest pair` stat: a
  small value means two picks are redundant with each other.
- If you need k=8 to be the first half of k=16, use `--no-refine`; swaps can
  replace a pick, so refined sets at different k are not prefixes.
- If it looks like a random grab bag, lower `--image-weight`; the tag signal is
  what captures "same style, different content".
- Check the tag coverage table first. A tag missing from the selection is the
  concrete lever for a feature that will otherwise dominate.
- Check the aspect table next. If one shape owns the corpus and the picks all
  follow it, the LoRA learns that composition too. Raise `--aspect-weight` to
  pull picks toward other shapes, or accept it and add images of other shapes
  to the source set: the picker can only rebalance what exists.
- Do not chase pixel-count variety. Above the training resolution it changes
  nothing after resampling; below it, the report flags the images instead.
- Select a subset, not the whole dataset: keep `-k` around 20-40 for a set of a
  few hundred images, and use `num_repeats` during training to hit your target
  step count. The images you leave out still exist if you need them.

## AI disclosure

This tool was written with an AI coding agent (DeepSeek Harness, running a
DeepSeek model). The agent produced most of the code, tests and documentation
in this repository; the human owner set the requirements, chose the scoring
design, reviewed the picks against real image sets, and reported the bugs that
shaped the current behaviour. Treat the code as agent-authored: read it before
relying on it, and expect an occasional rough edge.

## Layout

```
uniqpick/            the package (captions, features, scoring, select, report, cli)
tests/               unittest suite
install.sh           .venv + CPU torch + dependencies
run.sh               entry point, sets PYTHONPATH and the HF cache env
requirements.txt     pinned dependencies
AGENTS.md            notes for coding agents working on the tool
```

## Tests

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
```
