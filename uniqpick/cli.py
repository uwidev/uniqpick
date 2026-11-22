"""uniqpick command line: pick the most unique N images out of a folder."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import report, runstore
from .captions import caption_paths, tag_frequencies
from .data import Record, dedupe, discover
from .features import make_backend
from .scoring import aspect_distance, combine, image_distance, tag_distance, uniqueness_scores
from .select import METHODS, select

DEFAULT_MODEL = "hf-hub:timm/ViT-B-16-SigLIP2"
DEFAULT_SUFFIXES = (".txt", ".personal.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uniqpick",
        description=(
            "Pick the N images that best represent a varied sample of a folder: "
            "farthest-point / coverage selection over tag and image distance."
        ),
        epilog=(
            "Two-stage use: `uniqpick compute <folders> -o run/` computes the distances once, "
            "then `uniqpick pick run/ -k 24 -o out/` re-picks in milliseconds for any k, method "
            "or weight. This one-shot form does both steps in one go and also leaves the run "
            "files in the output folder, so a later `uniqpick pick` can reuse them."
        ),
    )
    parser.add_argument("input", nargs="+", type=Path, help="image folder(s) to read")
    parser.add_argument("-k", "--count", type=int, default=24, help="how many images to pick (default 24)")
    parser.add_argument("-o", "--out", type=Path, default=None, help="output folder (default: next to the input)")
    parser.add_argument("--method", choices=METHODS, default="coverage",
                        help="coverage spreads picks over the corpus (default); kcenter maximizes the "
                             "closest pair; disparity maximizes the summed pairwise distance")
    parser.add_argument("--no-refine", dest="refine", action="store_false",
                        help="keep the raw greedy order instead of improving it with swap search")
    parser.add_argument("--tag-weight", type=float, default=1.0)
    parser.add_argument("--image-weight", type=float, default=1.0)
    parser.add_argument("--aspect-weight", type=float, default=0.3,
                        help="amount frame shape adds to the tag and image distances, 0 disables "
                             "(default 0.3); one doubling of the aspect ratio counts as 1")
    parser.add_argument("--train-resolution", type=int, default=1024, metavar="RES",
                        help="training resolution, used by the report to flag images that "
                             "will be upscaled (default 1024)")
    parser.add_argument("--embeddings", choices=("auto", "clip", "basic", "none"), default="auto",
                        help="image features: clip/SigLIP (best), basic colour+layout, or none")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="open_clip model id")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exclude-tag", action="append", default=[], metavar="TAG",
                        help="drop this tag before scoring (repeatable)")
    parser.add_argument("--dedupe-hamming", type=int, default=6,
                        help="dhash bits under which two images count as near duplicates (-1 disables)")
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-df", type=float, default=0.95,
                        help="ignore tags present in more than this fraction of captions")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--caption-suffix", action="append", default=None, metavar=".txt",
                        help="caption suffix to read (repeatable; default .txt and .personal.txt)")
    parser.add_argument("--link", action="store_true", help="symlink into selected/ instead of copying")
    parser.add_argument("--no-cache", action="store_true", help="ignore cached embeddings")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug", action="store_true", help="re-raise errors with a traceback")
    return parser


def _add_weight_args(parser: argparse.ArgumentParser, *, defaults: bool) -> None:
    """Weights and training resolution: defaults for compute, overrides for pick."""
    if defaults:
        parser.add_argument("--tag-weight", type=float, default=1.0)
        parser.add_argument("--image-weight", type=float, default=1.0)
        parser.add_argument("--aspect-weight", type=float, default=0.3,
                            help="amount frame shape adds to the tag and image distances (default 0.3)")
        parser.add_argument("--train-resolution", type=int, default=1024, metavar="RES",
                            help="training resolution used by the report (default 1024)")
    else:
        parser.add_argument("--tag-weight", type=float, default=None,
                            help="override the tag weight recorded by compute")
        parser.add_argument("--image-weight", type=float, default=None,
                            help="override the image weight recorded by compute")
        parser.add_argument("--aspect-weight", type=float, default=None,
                            help="override the frame-shape weight recorded by compute")
        parser.add_argument("--train-resolution", type=int, default=None, metavar="RES",
                            help="override the training resolution recorded by compute")


def build_compute_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uniqpick compute",
        description=(
            "Compute the tag, image and frame-shape distance matrices for a folder and store "
            "them in a run folder. Reusable by `uniqpick pick` for any k, method or weight."
        ),
    )
    parser.add_argument("input", nargs="+", type=Path, help="image folder(s) to read")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="run folder (default: <input>-run next to the input)")
    parser.add_argument("--embeddings", choices=("auto", "clip", "basic", "none"), default="auto",
                        help="image features: clip/SigLIP (best), basic colour+layout, or none")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="open_clip model id")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exclude-tag", action="append", default=[], metavar="TAG",
                        help="drop this tag before scoring (repeatable)")
    parser.add_argument("--dedupe-hamming", type=int, default=6,
                        help="dhash bits under which two images count as near duplicates (-1 disables)")
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-df", type=float, default=0.95,
                        help="ignore tags present in more than this fraction of captions")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--caption-suffix", action="append", default=None, metavar=".txt",
                        help="caption suffix to read (repeatable; default .txt and .personal.txt)")
    _add_weight_args(parser, defaults=True)
    parser.add_argument("--no-cache", action="store_true", help="ignore cached embeddings")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug", action="store_true", help="re-raise errors with a traceback")
    return parser


def build_pick_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uniqpick pick",
        description=(
            "Pick N images from a run folder written by `uniqpick compute`. Only the cheap "
            "greedy walk and the report run again, so trying another k costs seconds."
        ),
    )
    parser.add_argument("run", type=Path, metavar="RUN", help="run folder written by `uniqpick compute`")
    parser.add_argument("-k", "--count", type=int, default=24, help="how many images to pick (default 24)")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="output folder (default: <run>/k<N>)")
    parser.add_argument("--method", choices=METHODS, default="coverage",
                        help="coverage spreads picks over the corpus (default); kcenter maximizes the "
                             "closest pair; disparity maximizes the summed pairwise distance")
    parser.add_argument("--no-refine", dest="refine", action="store_false",
                        help="keep the raw greedy order instead of improving it with swap search")
    _add_weight_args(parser, defaults=False)
    parser.add_argument("--exclude-tag", action="append", default=[], metavar="TAG",
                        help="drop this tag and rebuild the tag distance (repeatable)")
    parser.add_argument("--link", action="store_true", help="symlink into selected/ instead of copying")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug", action="store_true", help="re-raise errors with a traceback")
    return parser


def log(quiet: bool, message: str) -> None:
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def embedding_cache_path(out_dir: Path, backend: str, model: str, records: list[Record]) -> Path:
    # Key on the content hash, not size/mtime: replacing an image with another
    # one that happens to keep its timestamp (cp -p, git checkout) would
    # otherwise silently reuse embeddings of the old picture.
    digest = hashlib.md5()
    for record in records:
        digest.update(f"{record.path}|{record.md5}|".encode())
    slug = "".join(ch if ch.isalnum() else "-" for ch in model)[-48:]
    return out_dir / ".cache" / f"emb-{backend}-{slug}-{digest.hexdigest()[:16]}.npy"


def load_embeddings(args, out_dir: Path, records: list[Record]) -> tuple[np.ndarray | None, object | None]:
    """Return (embeddings, backend) or (None, None) when image features are off."""
    if args.embeddings == "none":
        return None, None

    kind = args.embeddings
    backend = None
    if kind == "auto":
        try:
            backend = make_backend("clip", args.model, args.device)
        except Exception as exc:  # missing torch, gated weights, no network, ...
            log(args.quiet, f"uniqpick: clip backend unavailable ({exc}); using basic colour features")
            backend = make_backend("basic", args.model, args.device)
    else:
        backend = make_backend(kind, args.model, args.device)

    cache = embedding_cache_path(out_dir, backend.name, args.model, records)
    if not args.no_cache and cache.exists():
        cached = np.load(cache)
        if cached.shape[0] == len(records):
            log(args.quiet, f"uniqpick: embeddings from cache ({cached.shape[1]}-d)")
            return cached, backend

    paths = [record.path for record in records]
    log(args.quiet, f"uniqpick: embedding {len(paths)} images with {backend.name} on {args.device}")
    done = {"n": -1}

    def progress(current: int, total: int) -> None:
        step = max(1, total // 10)
        if current == total or current - done["n"] >= step:
            done["n"] = current
            log(args.quiet, f"  {current}/{total}")

    embeddings = backend.embed(paths, progress=progress)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, embeddings)
    return embeddings, backend


GENERATED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".txt"}


def clear_generated(folder: Path) -> None:
    """Remove previous outputs of this tool so a re-run cannot mix two selections.

    Only file types uniqpick writes are removed; anything else in the folder is
    left alone.
    """
    if not folder.is_dir():
        return
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in GENERATED_SUFFIXES:
            path.unlink()


def copy_selection(out_dir: Path, picked: list[Record], suffixes: tuple[str, ...], link: bool) -> Path:
    target = out_dir / "selected"
    target.mkdir(parents=True, exist_ok=True)
    clear_generated(target)
    used: set[str] = set()

    for record in picked:
        name = record.path.name
        counter = 1
        while name.lower() in used:
            counter += 1
            name = f"{record.path.stem}-{counter}{record.path.suffix}"
        used.add(name.lower())
        destination = target / name

        if destination.exists() or destination.is_symlink():
            destination.unlink()
        if link:
            destination.symlink_to(record.path.resolve())
        else:
            shutil.copy2(record.path, destination)

        for caption in caption_paths(record.path, suffixes):
            if not caption.exists():
                continue
            side_name = Path(name).stem + caption.name[len(record.path.stem):]
            side_destination = target / side_name
            if side_destination.exists() or side_destination.is_symlink():
                side_destination.unlink()
            if link:
                side_destination.symlink_to(caption.resolve())
            else:
                shutil.copy2(caption, side_destination)

    return target


def resolve_inputs(inputs: list[Path]) -> list[Path]:
    roots = [Path(root).expanduser().resolve() for root in inputs]
    for root in roots:
        if not root.is_dir():
            raise SystemExit(f"uniqpick: not a folder: {root}")
    return roots


def build_run(args, roots: list[Path], run_dir: Path) -> runstore.RunData:
    """Discover, dedupe and measure a corpus, then persist the result into ``run_dir``."""
    suffixes = tuple(args.caption_suffix) if args.caption_suffix else DEFAULT_SUFFIXES
    excluded = tuple(args.exclude_tag)

    t0 = time.time()
    records = discover(roots, recursive=args.recursive, caption_suffixes=suffixes, extra_exclude=excluded)
    records = [record for record in records if run_dir not in record.path.parents]
    if not records:
        raise SystemExit(f"uniqpick: no images found in {', '.join(str(root) for root in roots)}")
    log(args.quiet, f"uniqpick: found {len(records)} images ({time.time() - t0:.1f}s)")

    kept = dedupe(records, max_hamming=args.dedupe_hamming)
    dropped = len(records) - len(kept)
    if dropped:
        log(args.quiet, f"uniqpick: dropped {dropped} duplicates, {len(kept)} remain")

    # Tag distance -----------------------------------------------------------
    docs = [record.tags for record in kept]
    tag_dist, vocab = (None, [])
    if any(docs):
        t0 = time.time()
        tag_dist, vocab = tag_distance(docs, min_df=args.min_df, max_df=args.max_df)
        log(args.quiet, f"uniqpick: tag space {len(vocab)} tags ({time.time() - t0:.1f}s)")
    else:
        log(args.quiet, "uniqpick: no captions found, scoring on image features only")

    # Image distance ---------------------------------------------------------
    embeddings, backend = load_embeddings(args, run_dir, kept)
    image_dist = None
    if embeddings is not None and backend is not None:
        image_dist = image_distance(embeddings, standardize=bool(getattr(backend, "standardize", False)))

    if tag_dist is None and image_dist is None:
        raise SystemExit("uniqpick: no usable signal (no captions and image features disabled)")

    # Frame shape is a soft third signal: it does not decide the pick on its
    # own, it makes the distance report varied framing as actual variety.
    ratios = np.array([record.width / record.height for record in kept], dtype=np.float32)
    aspect_dist = aspect_distance(ratios)

    index_of = {id(record): index for index, record in enumerate(records)}
    meta = {
        "version": runstore.RUN_VERSION,
        "tool": "uniqpick",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": [str(root) for root in roots],
        "backend": backend.name if backend is not None else "none",
        "model": args.model,
        "device": args.device,
        "dedupe_hamming": args.dedupe_hamming,
        "caption_suffixes": list(suffixes),
        "extra_exclude": list(excluded),
        "min_df": args.min_df,
        "max_df": args.max_df,
        "train_resolution": args.train_resolution,
        "tag_weight": args.tag_weight,
        "image_weight": args.image_weight,
        "aspect_weight": args.aspect_weight,
        "tag_vocab": list(vocab),
        "tag_freq": tag_frequencies(docs),
        "records": runstore.encode_records(records),
        "kept": [index_of[id(record)] for record in kept],
    }
    runstore.save_run(run_dir, meta, tag_dist, image_dist, aspect_dist)

    return runstore.RunData(
        meta=meta,
        records=records,
        kept=kept,
        tag_dist=tag_dist,
        image_dist=image_dist,
        aspect_dist=aspect_dist,
    )


@dataclass
class PickOptions:
    """Everything that can change without recomputing the distances."""

    count: int
    method: str = "coverage"
    tag_weight: float | None = None
    image_weight: float | None = None
    aspect_weight: float | None = None
    train_resolution: int | None = None
    exclude_tags: tuple[str, ...] = ()
    link: bool = False
    refine: bool = True
    quiet: bool = False


def finish(data: runstore.RunData, opts: PickOptions, out_dir: Path) -> int:
    """Select, copy and report from already computed distances."""
    started = time.time()
    meta = data.meta
    kept = data.kept
    records = data.records

    tag_dist = data.tag_dist
    vocab = list(meta.get("tag_vocab", []))
    tag_freq = dict(meta.get("tag_freq", {}))
    if opts.exclude_tags:
        if tag_dist is None:
            raise SystemExit("uniqpick: --exclude-tag needs a run computed with captions")
        docs = [
            {tag: weight for tag, weight in record.tags.items() if tag not in opts.exclude_tags}
            for record in kept
        ]
        tag_dist, vocab = tag_distance(
            docs,
            min_df=int(meta.get("min_df", 1)),
            max_df=float(meta.get("max_df", 0.95)),
        )
        tag_freq = tag_frequencies(docs)

    image_dist = data.image_dist
    if tag_dist is None and image_dist is None:
        raise SystemExit("uniqpick: no usable signal (no captions and image features disabled)")

    count = min(opts.count, len(kept))
    if count < opts.count:
        log(opts.quiet, f"uniqpick: only {len(kept)} images available, picking all of them")

    ratios = np.array([record.width / record.height for record in kept], dtype=np.float32)
    aspect_dist = data.aspect_dist if data.aspect_dist is not None else aspect_distance(ratios)

    tag_weight = opts.tag_weight if opts.tag_weight is not None else float(meta.get("tag_weight", 1.0))
    image_weight = (
        opts.image_weight if opts.image_weight is not None else float(meta.get("image_weight", 1.0))
    )
    aspect_weight = (
        opts.aspect_weight if opts.aspect_weight is not None else float(meta.get("aspect_weight", 0.3))
    )
    train_resolution = (
        opts.train_resolution
        if opts.train_resolution is not None
        else int(meta.get("train_resolution", 1024))
    )

    distance, applied_tag, applied_image, applied_aspect = combine(
        tag_dist,
        image_dist,
        tag_weight,
        image_weight,
        aspect_dist=aspect_dist,
        aspect_weight=aspect_weight,
    )
    uniqueness = uniqueness_scores(distance)
    selection = select(distance, count, method=opts.method, refine=opts.refine)
    # The same pick on one signal at a time, so the report can show which half of
    # the score drives each choice.
    selection_tag = (
        select(tag_dist, count, method=opts.method, refine=opts.refine)
        if tag_dist is not None else None
    )
    selection_image = (
        select(image_dist, count, method=opts.method, refine=opts.refine)
        if image_dist is not None else None
    )

    picked = [kept[index] for index in selection.order]
    suffixes = tuple(meta.get("caption_suffixes") or DEFAULT_SUFFIXES)
    chosen = copy_selection(out_dir, picked, suffixes, opts.link)

    ctx = report.ReportContext(
        out_dir=out_dir,
        input_roots=[Path(root) for root in meta.get("inputs", [])],
        records=records,
        kept=kept,
        selection=selection,
        distance=distance,
        uniqueness=uniqueness,
        tag_vocab=vocab,
        tag_freq=tag_freq,
        backend=meta.get("backend", "none"),
        tag_weight=applied_tag,
        image_weight=applied_image,
        dedupe_hamming=int(meta.get("dedupe_hamming", 6)),
        thumb_dir=out_dir / "thumbs",
        selection_tag=selection_tag,
        selection_image=selection_image,
        distance_tag=tag_dist,
        distance_image=image_dist,
        aspect_weight=applied_aspect,
        aspect_ratios=ratios,
        aspect_distance=aspect_dist,
        train_resolution=train_resolution,
        requested=opts.count,
    )

    report.clear_generated(ctx.thumb_dir)
    names = report.make_thumbnails(ctx)
    report.write_csv(ctx, out_dir / "selection.csv")
    report.write_markdown(ctx, out_dir / "report.md")
    report.write_html(ctx, names, out_dir / "report.html")
    report.write_json(ctx, names, out_dir / "selection.json")
    sheet = report.contact_sheet(ctx, names, out_dir / "selected_sheet.png")
    if selection_tag is not None:
        report.contact_sheet(ctx, names, out_dir / "selected_tags_sheet.png", selection=selection_tag)
    if selection_image is not None:
        report.contact_sheet(ctx, names, out_dir / "selected_images_sheet.png", selection=selection_image)

    combined = set(selection.order)
    overlap_lines = []
    if selection_tag is not None:
        overlap_lines.append(
            f"  tags only  {len(set(selection_tag.order) & combined)}/{count} shared with tags+image"
        )
    if selection_image is not None:
        overlap_lines.append(
            f"  image only {len(set(selection_image.order) & combined)}/{count} shared with tags+image"
        )

    elapsed = time.time() - started
    print(
        f"uniqpick: {len(records)} images -> {len(kept)} after dedupe -> {count} selected\n"
        f"  method {selection.method}{' + swap' if selection.refined else ''}"
        f" (objective {selection.objective:.4f}),"
        f" weights tag {applied_tag:.2f} / image {applied_image:.2f},"
        f" + aspect {applied_aspect:.2f}, features {ctx.backend} ({len(vocab)} tags)\n"
        + "".join(line + "\n" for line in overlap_lines)
        + f"  selected  {chosen}\n"
        f"  report    {out_dir / 'report.html'}\n"
        f"  sheet     {sheet}\n"
        f"  done in {elapsed:.1f}s"
    )
    return 0


def run(args: argparse.Namespace) -> int:
    """One-shot form: compute the run files and pick in the same output folder."""
    roots = resolve_inputs(args.input)
    if args.count < 1:
        raise SystemExit("uniqpick: --count must be at least 1")

    out_dir = Path(args.out).expanduser().resolve() if args.out else roots[0].parent / f"{roots[0].name}-uniqpick"
    data = build_run(args, roots, out_dir)
    opts = PickOptions(
        count=args.count,
        method=args.method,
        tag_weight=args.tag_weight,
        image_weight=args.image_weight,
        aspect_weight=args.aspect_weight,
        train_resolution=args.train_resolution,
        link=args.link,
        refine=args.refine,
        quiet=args.quiet,
    )
    code = finish(data, opts, out_dir)
    if not args.quiet:
        print(f"  run       {out_dir} (reuse with: uniqpick pick {out_dir} -k {args.count})", file=sys.stderr)
    return code


def compute_command(args: argparse.Namespace) -> int:
    started = time.time()
    roots = resolve_inputs(args.input)
    run_dir = Path(args.out).expanduser().resolve() if args.out else roots[0].parent / f"{roots[0].name}-run"

    data = build_run(args, roots, run_dir)
    elapsed = time.time() - started
    print(
        f"uniqpick: computed {len(data.records)} images -> {len(data.kept)} after dedupe\n"
        f"  run       {run_dir}\n"
        f"  features  {data.meta['backend']} ({len(data.meta['tag_vocab'])} tags)\n"
        f"  next      uniqpick pick {run_dir} -k 24\n"
        f"  done in {elapsed:.1f}s"
    )
    return 0


def pick_command(args: argparse.Namespace) -> int:
    if args.count < 1:
        raise SystemExit("uniqpick: --count must be at least 1")

    run_dir = Path(args.run).expanduser().resolve()
    data = runstore.load_run(run_dir)

    if args.out:
        out_dir = Path(args.out).expanduser().resolve()
    else:
        suffix = "" if args.method == "coverage" else f"-{args.method}"
        out_dir = run_dir / f"k{args.count}{suffix}"

    opts = PickOptions(
        count=args.count,
        method=args.method,
        tag_weight=args.tag_weight,
        image_weight=args.image_weight,
        aspect_weight=args.aspect_weight,
        train_resolution=args.train_resolution,
        exclude_tags=tuple(args.exclude_tag),
        link=args.link,
        refine=args.refine,
        quiet=args.quiet,
    )
    log(args.quiet, f"uniqpick: picking from {run_dir}")
    return finish(data, opts, out_dir)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv and argv[0] in ("compute", "pick") else None

    if command == "compute":
        args = build_compute_parser().parse_args(argv[1:])
        handler = compute_command
    elif command == "pick":
        args = build_pick_parser().parse_args(argv[1:])
        handler = pick_command
    else:
        args = build_parser().parse_args(argv)
        handler = run

    try:
        return handler(args)
    except SystemExit:
        raise
    except runstore.RunFormatError as exc:
        print(f"uniqpick: error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        if args.debug:
            raise
        print(f"uniqpick: error: {exc}", file=sys.stderr)
        return 1
