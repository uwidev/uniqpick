"""Human-facing output: CSV, Markdown, HTML gallery and contact sheets."""

from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import Record
from .features import open_rgb
from .scoring import ASPECT_BUCKETS, aspect_bucket
from .select import Selection, kcenter_objective, nearest_selected

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
)


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


@dataclass
class ReportContext:
    out_dir: Path
    input_roots: list[Path]
    records: list[Record]
    kept: list[Record]
    selection: Selection
    distance: object  # np.ndarray, typed loosely to keep numpy out of this module
    uniqueness: object
    tag_vocab: list[str]
    tag_freq: dict[str, int]
    backend: str
    tag_weight: float
    image_weight: float
    dedupe_hamming: int
    thumb_dir: Path
    # The same pick run on one signal at a time, so the report can show what each
    # half of the score contributes. None when that signal is unavailable.
    selection_tag: Selection | None = None
    selection_image: Selection | None = None
    distance_tag: object | None = None
    distance_image: object | None = None
    # Frame shape: a soft third signal, reported separately because it answers a
    # different question than "which picture is this" (see scoring.aspect_distance).
    aspect_weight: float = 0.0
    aspect_ratios: object | None = None
    aspect_distance: object | None = None
    train_resolution: int = 1024
    # How many picks the user asked for; differs from len(selection.order) when
    # the folder held fewer images than requested.
    requested: int = 0

    def views(self) -> list[tuple[str, str, Selection | None, object | None]]:
        """(key, heading, selection, distance) for every available view."""
        rows: list[tuple[str, str, Selection | None, object | None]] = [
            ("combined", "tags + image", self.selection, self.distance),
        ]
        if self.selection_tag is not None:
            rows.append(("tags", "tags only", self.selection_tag, self.distance_tag))
        if self.selection_image is not None:
            rows.append(("images", "image only", self.selection_image, self.distance_image))
        return rows


def record_index(ctx: ReportContext) -> dict[int, int]:
    return {id(record): i for i, record in enumerate(ctx.records)}


def thumb_name(index: int, record: Record) -> str:
    return f"{index:04d}_{record.path.stem[:16]}.jpg"


def clear_generated(folder: Path) -> None:
    """Drop thumbnails from an earlier run so the report never shows stale files."""
    if not folder.is_dir():
        return
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            path.unlink()


def make_thumbnails(ctx: ReportContext, max_side: int = 420) -> dict[int, str]:
    """Render small copies so the report stays portable and opens instantly."""
    ctx.thumb_dir.mkdir(parents=True, exist_ok=True)
    names: dict[int, str] = {}
    for index, record in enumerate(ctx.records):
        name = thumb_name(index, record)
        target = ctx.thumb_dir / name
        if not target.exists():
            image = open_rgb(record.path, max_side=max_side)
            image.save(target, "JPEG", quality=86)
        names[id(record)] = name
    return names


def tag_list(record: Record, vocab: set[str], limit: int = 12) -> str:
    """Only discriminating tags: the constant artist and rating tags are noise here."""
    tags = [
        (tag, weight)
        for tag, weight in sorted(record.tags.items(), key=lambda item: (-item[1], item[0]))
        if tag in vocab
    ]
    shown = [tag for tag, _ in tags[:limit]]
    text = ", ".join(html.escape(tag) for tag in shown)
    if len(tags) > limit:
        text += f' <span class="more">+{len(tags) - limit}</span>'
    return text


def contact_sheet(
    ctx: ReportContext,
    names: dict[int, str],
    output: Path,
    cols: int = 6,
    tile: int = 300,
    selection: Selection | None = None,
) -> Path:
    """Numbered grid of the selected images, in pick order."""
    chosen = (selection or ctx.selection).order
    rows = (len(chosen) + cols - 1) // cols
    pad, label_h = 6, 34
    cell_w, cell_h = tile + pad * 2, tile + pad * 2 + label_h
    sheet = Image.new("RGB", (cols * cell_w, max(rows, 1) * cell_h), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    font = _font(22)

    for position, record_index in enumerate(chosen):
        record = ctx.kept[record_index]
        image = open_rgb(record.path, max_side=tile)
        x = (position % cols) * cell_w + pad
        y = (position // cols) * cell_h + pad
        sheet.paste(image, (x, y))
        draw.text((x + 2, y + tile + 4), f"{position + 1}", fill=(255, 214, 84), font=font)

    sheet.save(output, "PNG")
    return output


def write_csv(ctx: ReportContext, output: Path) -> Path:
    chosen = set(ctx.selection.order)
    tag_chosen = set(ctx.selection_tag.order) if ctx.selection_tag else set()
    image_chosen = set(ctx.selection_image.order) if ctx.selection_image else set()
    rank_of = {index: rank + 1 for rank, index in enumerate(ctx.selection.order)}
    tag_rank = {index: rank + 1 for rank, index in enumerate(ctx.selection_tag.order)} if ctx.selection_tag else {}
    image_rank = {index: rank + 1 for rank, index in enumerate(ctx.selection_image.order)} if ctx.selection_image else {}
    columns = [
        "file",
        "selected",
        "rank",
        "tag_selected",
        "tag_rank",
        "image_selected",
        "image_rank",
        "duplicate_of",
        "uniqueness",
        "pick_score",
        "min_dist_to_previous",
        "nearest_selected",
        "nearest_distance",
        "tag_distance",
        "image_distance",
        "aspect_ratio",
        "aspect_bucket",
        "aspect_distance",
        "tags",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, record in enumerate(ctx.kept):
            row = ctx.distance[index]
            nearest_pos = min(
                ctx.selection.order, key=lambda candidate: row[candidate]
            ) if chosen else None
            writer.writerow(
                {
                    "file": record.path.name,
                    "selected": int(index in chosen),
                    "rank": rank_of.get(index, ""),
                    "tag_selected": int(index in tag_chosen),
                    "tag_rank": tag_rank.get(index, ""),
                    "image_selected": int(index in image_chosen),
                    "image_rank": image_rank.get(index, ""),
                    "duplicate_of": record.duplicate_of.path.name if record.duplicate_of else "",
                    "uniqueness": f"{ctx.uniqueness[index]:.4f}",
                    "pick_score": (
                        f"{ctx.selection.scores[ctx.selection.order.index(index)]:.4f}"
                        if index in chosen
                        else ""
                    ),
                    "min_dist_to_previous": (
                        f"{ctx.selection.nearest[ctx.selection.order.index(index)]:.4f}"
                        if index in chosen
                        else ""
                    ),
                    "nearest_selected": ctx.kept[nearest_pos].path.name if nearest_pos is not None else "",
                    "nearest_distance": f"{row[nearest_pos]:.4f}" if nearest_pos is not None else "",
                    "tag_distance": (
                        f"{ctx.distance_tag[index][nearest_pos]:.4f}"
                        if nearest_pos is not None and ctx.distance_tag is not None
                        else ""
                    ),
                    "image_distance": (
                        f"{ctx.distance_image[index][nearest_pos]:.4f}"
                        if nearest_pos is not None and ctx.distance_image is not None
                        else ""
                    ),
                    "aspect_ratio": f"{record.width / record.height:.4f}",
                    "aspect_bucket": aspect_bucket(record.width / record.height),
                    "aspect_distance": (
                        f"{ctx.aspect_distance[index][nearest_pos]:.4f}"
                        if nearest_pos is not None and ctx.aspect_distance is not None
                        else ""
                    ),
                    "tags": ", ".join(sorted(record.tags)),
                }
            )
    return output


def _coverage_rows(ctx: ReportContext, limit: int = 36) -> list[tuple[str, int, int, str]]:
    total = len(ctx.kept)
    pick = len(ctx.selection.order)
    counts: dict[str, int] = {}
    for index in ctx.selection.order:
        for tag in ctx.kept[index].tags:
            counts[tag] = counts.get(tag, 0) + 1
    # Only tags that actually discriminate belong here: anything filtered out of
    # the TF-IDF space (the artist tag, the rating) is not a coverage lever.
    scoring_tags = set(ctx.tag_vocab)
    rows = []
    for tag, count in sorted(ctx.tag_freq.items(), key=lambda item: (-item[1], item[0])):
        if count < 2 or tag not in scoring_tags:
            continue
        picked = counts.get(tag, 0)
        expected = count * pick / max(total, 1)
        if picked == 0 and count >= 3:
            flag = "missing"
        elif expected >= 1 and picked / max(expected, 1e-9) >= 1.8:
            flag = "over"
        else:
            flag = ""
        rows.append((tag, count, picked, flag))
        if len(rows) >= limit:
            break
    return rows


def _aspect_rows(ctx: ReportContext) -> list[tuple[str, int, int, str]]:
    """Frame-shape buckets: corpus count, picked count, and the same flags as tags.

    A bucket with three or more images and no pick is where a shape can dominate:
    the style then only exists in that framing.
    """
    total = len(ctx.kept)
    pick = len(ctx.selection.order)
    chosen = set(ctx.selection.order)
    counts: dict[str, int] = {}
    picked: dict[str, int] = {}
    for index, ratio in enumerate(ctx.aspect_ratios if ctx.aspect_ratios is not None else []):
        bucket = aspect_bucket(float(ratio))
        counts[bucket] = counts.get(bucket, 0) + 1
        if index in chosen:
            picked[bucket] = picked.get(bucket, 0) + 1
    rows = []
    for _ceiling, bucket in ASPECT_BUCKETS:
        count = counts.get(bucket, 0)
        if not count:
            continue
        hits = picked.get(bucket, 0)
        expected = count * pick / max(total, 1)
        if hits == 0 and count >= 3:
            flag = "missing"
        elif hits == 0:
            flag = "thin"
        elif expected >= 1 and hits / max(expected, 1e-9) >= 1.8:
            flag = "over"
        else:
            flag = ""
        rows.append((bucket, count, hits, flag))
    return rows


def _short_side(values: list[int]) -> tuple[int, int, int]:
    ordered = sorted(values)
    if not ordered:
        return (0, 0, 0)
    return (ordered[0], ordered[len(ordered) // 2], ordered[-1])


def _size_stats(ctx: ReportContext) -> tuple[tuple[int, int, int], tuple[int, int, int], list[Record]]:
    """Short-side min/median/max for corpus and picks, plus images below the training size."""
    corpus = [min(record.width, record.height) for record in ctx.kept]
    picks = [min(ctx.kept[index].width, ctx.kept[index].height) for index in ctx.selection.order]
    small = [record for record in ctx.kept if min(record.width, record.height) < ctx.train_resolution]
    return _short_side(corpus), _short_side(picks), small


def _aspect_label(ctx: ReportContext, index: int) -> str:
    ratio = float(ctx.aspect_ratios[index]) if ctx.aspect_ratios is not None else 1.0
    return f"{aspect_bucket(ratio)} {ratio:.2f}"


def write_markdown(ctx: ReportContext, output: Path) -> Path:
    total = len(ctx.records)
    kept = len(ctx.kept)
    picked = len(ctx.selection.order)
    duplicates = total - kept
    lines = [
        "# uniqpick report",
        "",
        f"- input: {', '.join(str(root) for root in ctx.input_roots)}",
        f"- images found: {total}, after dedupe: {kept}, selected: {picked}"
        + (f" (asked for {ctx.requested})" if ctx.requested > picked else ""),
        f"- method: {ctx.selection.method}, tag weight {ctx.tag_weight:.2f}, "
        f"image weight {ctx.image_weight:.2f}, aspect weight +{ctx.aspect_weight:.2f}",
        f"- image features: {ctx.backend}, tag vocabulary: {len(ctx.tag_vocab)}",
        f"- duplicate threshold: {ctx.dedupe_hamming} bits (negative disables near-dup checks)",
        "",
        "## Selection (pick order)",
        "",
        "| # | file | uniqueness | min dist to previous |",
        "|---|------|-----------:|---------------------:|",
    ]
    for rank, index in enumerate(ctx.selection.order, start=1):
        record = ctx.kept[index]
        lines.append(
            f"| {rank} | {record.path.name} | {ctx.uniqueness[index]:.3f} | "
            f"{ctx.selection.nearest[rank - 1]:.3f} |"
        )

    rows = _coverage_rows(ctx)
    if rows:
        lines += [
            "",
            "## Tag coverage",
            "",
            "`missing` = present in 3+ images but never picked. `over` = picked at "
            "roughly twice its corpus share.",
            "",
            "| tag | images | picked | note |",
            "|-----|-------:|-------:|------|",
        ]
        for tag, count, picked_count, flag in rows:
            lines.append(f"| {tag} | {count} | {picked_count} | {flag} |")

    aspect_rows = _aspect_rows(ctx)
    if aspect_rows:
        corpus_short, picks_short, small = _size_stats(ctx)
        lines += [
            "",
            "## Aspect and size",
            "",
            f"Short side: corpus {corpus_short[0]} / {corpus_short[1]} / {corpus_short[2]} px, "
            f"picks {picks_short[0]} / {picks_short[1]} / {picks_short[2]} px "
            f"(min / median / max). Frame shape adds up to {ctx.aspect_weight:.2f} to the "
            "tag and image sums; the final distance is divided by the total weight, "
            "so it still reads 0..1.",
            "",
            "`missing` = 3+ images of that shape, none picked. `thin` = only 1-2 "
            "images of that shape, none picked: too rare to force a pick, so the "
            "weight cannot help. Aspect survives training resampling; pixel count "
            "does not.",
            "",
            "| bucket | images | picked | note |",
            "|--------|-------:|-------:|------|",
        ]
        for bucket, count, hits, flag in aspect_rows:
            lines.append(f"| {bucket} | {count} | {hits} | {flag} |")
        if small:
            names = ", ".join(
                f"{record.path.name} ({record.width}x{record.height})" for record in small[:10]
            )
            more = f" and {len(small) - 10} more" if len(small) > 10 else ""
            lines += [
                "",
                f"{len(small)} images are below {ctx.train_resolution} px on the short side "
                f"and would be upscaled: {names}{more}",
            ]

    for key, heading, selection, _distance in ctx.views():
        if key == "combined" or selection is None:
            continue
        lines += [
            "",
            f"## Selection from {heading} (pick order)",
            "",
            "| # | file | in tags+image set |",
            "|---|------|------------------|",
        ]
        combined = set(ctx.selection.order)
        for rank, index in enumerate(selection.order, start=1):
            record = ctx.kept[index]
            shared = "yes" if index in combined else "no"
            lines.append(f"| {rank} | {record.path.name} | {shared} |")

    if duplicates:
        lines += ["", "## Dropped as duplicates", ""]
        for record in ctx.records:
            if record.duplicate_of is not None:
                lines.append(
                    f"- {record.path.name} ({record.dup_reason}) -> kept "
                    f"{record.duplicate_of.path.name}"
                )

    lines += [
        "",
        "## Notes",
        "",
        "- The first pick is the corpus medoid (most representative image); later picks add coverage.",
        "- `uniqueness` is mean distance to all other kept images, independent of the pick order.",
        "- Methods: `coverage` (default) spreads picks over the corpus, `kcenter` maximizes the "
        "closest pair, `disparity` maximizes the summed pairwise distance.",
        "- After the greedy walk every method runs a swap search: a pick and an unpicked image "
        "trade places while the objective rises (`--no-refine` keeps the raw greedy order).",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def won_against(
    distance: np.ndarray, selection: Selection, limit: int = 0
) -> dict[int, list[tuple[int, float]]]:
    """Picks and the images each one stands in for.

    Every image that is not a pick has a nearest pick; that pick is why the
    image was skipped, so the pick "won against" it.  Groups are sorted by
    distance, closest first, so the thumbnails read as "almost a duplicate"
    through "nearly made the cut".
    """
    chosen = list(selection.order)
    chosen_set = set(chosen)
    nearest, gap = nearest_selected(distance, chosen)
    groups: dict[int, list[tuple[int, float]]] = {index: [] for index in chosen}
    for row in range(distance.shape[0]):
        if row in chosen_set:
            continue
        groups[int(nearest[row])].append((row, float(gap[row])))
    for items in groups.values():
        items.sort(key=lambda item: item[1])
        if limit:
            del items[limit:]
    return groups


def write_html(ctx: ReportContext, names: dict[int, str], output: Path) -> Path:
    total = len(ctx.records)
    kept = len(ctx.kept)
    picked = len(ctx.selection.order)
    short_note = (
        f'<p class="note"><b>{ctx.requested}</b> picks requested, but the folder holds '
        f'{kept} images after dedupe, so every image was picked.</p>'
        if ctx.requested > picked
        else ""
    )
    combined = set(ctx.selection.order)
    combined_rank = {index: rank + 1 for rank, index in enumerate(ctx.selection.order)}
    tag_rank = {index: rank + 1 for rank, index in enumerate(ctx.selection_tag.order)} if ctx.selection_tag else {}
    image_rank = {index: rank + 1 for rank, index in enumerate(ctx.selection_image.order)} if ctx.selection_image else {}
    scoring_tags = set(ctx.tag_vocab)

    def fmt(value: float) -> str:
        return f"{value:.3f}"

    def figure(record: Record, shot: str, figure_class: str = "card") -> str:
        # The shot is not wrapped in a single link: the hover overlays contain
        # links to other files, and anchors cannot be nested.
        return f"""
        <figure class="{figure_class}">
          {shot}
          <figcaption>
            <a class="fn" href="{quote(str(record.path))}" target="_blank"
               title="open {html.escape(record.path.name)}">{html.escape(record.path.name)}</a>
          </figcaption>
        </figure>"""

    def won_html(groups: dict[int, list[tuple[int, float]]], index: int) -> str:
        losers = groups.get(index, [])
        if not losers:
            return (
                '<div class="won"><div class="wonhead">'
                '<b>won against 0</b> <span class="dim">no image has this as nearest pick</span>'
                "</div></div>"
            )
        strip = []
        shown = losers[:8]
        for row, gap in shown:
            other = ctx.kept[row]
            strip.append(
                f'<a href="{quote(str(other.path))}" target="_blank" '
                f'title="open {html.escape(other.path.name)}">'
                f'<img src="{quote("thumbs/" + names[id(other)])}" alt=""></a>'
            )
        extra = len(losers) - len(shown)
        more = f'<span class="more">+{extra}</span>' if extra > 0 else ""
        span = (
            f"closest {fmt(losers[0][1])}" if len(losers) == 1
            else f"closest {fmt(losers[0][1])} to {fmt(losers[-1][1])}"
        )
        return (
            f'<div class="won"><div class="wonhead"><b>won against {len(losers)}</b>'
            f' <span class="dim">{span}</span></div>'
            f'<div class="strip">{"".join(strip)}{more}</div></div>'
        )

    def card(
        index: int,
        selection: Selection,
        view: str,
        distance: np.ndarray,
        groups: dict[int, list[tuple[int, float]]],
    ) -> str:
        record = ctx.kept[index]
        rank = selection.order.index(index) + 1
        tail = "medoid" if rank == 1 else f"min-dist {fmt(selection.nearest[rank - 1])}"
        membership = ""
        if view != "combined":
            if index in combined_rank:
                membership = f'<span class="pill shared">in tags+image #{combined_rank[index]}</span>'
            else:
                membership = '<span class="pill solo">not in tags+image</span>'
        body = (
            f'<div class="shot">'
            f'<a class="open" href="{quote(str(record.path))}" target="_blank">'
            f'<img loading="lazy" src="{quote("thumbs/" + names[id(record)])}" alt=""></a>'
            f'<span class="badge">{rank}</span>'
            f'{won_html(groups, index)}</div>'
        )
        figure_html = figure(record, body)
        metrics = (
            f'<div class="meta">{membership}unique {ctx.uniqueness[index]:.3f}'
            f' <span class="sep">|</span> {tail}'
            f' <span class="sep">|</span> {_aspect_label(ctx, index)}</div>'
            f'<div class="tags">{tag_list(record, scoring_tags)}</div>'
        )
        # Insert the two metric lines before </figcaption> so the shared card
        # helper keeps handling the link and the file name.
        return figure_html.replace("</figcaption>", metrics + "</figcaption>")

    def lost_card(index: int) -> str:
        record = ctx.kept[index]
        row = ctx.distance[index]
        nearest = min(ctx.selection.order, key=lambda candidate: row[candidate])
        rank = combined_rank[nearest]
        parts = [f"all {fmt(row[nearest])}"]
        if ctx.distance_tag is not None:
            parts.append(f"tag {fmt(ctx.distance_tag[index][nearest])}")
        if ctx.distance_image is not None:
            parts.append(f"image {fmt(ctx.distance_image[index][nearest])}")
        where = []
        if index in tag_rank:
            where.append(f"tags #{tag_rank[index]}")
        if index in image_rank:
            where.append(f"image #{image_rank[index]}")
        extra = (" &#183; " + " &#183; ".join(where)) if where else ""
        pick_thumb = quote("thumbs/" + names[id(ctx.kept[nearest])])
        pick_path = quote(str(ctx.kept[nearest].path))
        lost = (
            f'<a class="lost" href="{pick_path}" target="_blank" '
            f'title="open {html.escape(ctx.kept[nearest].path.name)}">'
            f'<img src="{pick_thumb}" alt="">'
            f'<div class="losttext"><b>lost to #{rank}</b>'
            f'<div class="dim name">{html.escape(ctx.kept[nearest].path.name)}</div>'
            f'<div>{" &#183; ".join(parts)}{extra}</div></div>'
            f'</a>'
        )
        body = (
            f'<div class="shot">'
            f'<a class="open" href="{quote(str(record.path))}" target="_blank">'
            f'<img loading="lazy" src="{quote("thumbs/" + names[id(record)])}" alt=""></a>'
            f'{lost}</div>'
        )
        figure_html = figure(record, body, "card dim")
        metrics = (
            f'<div class="meta">unique {ctx.uniqueness[index]:.3f}'
            f' <span class="sep">|</span> nearest #{rank} ({fmt(row[nearest])})</div>'
            f'<div class="tags">{tag_list(record, scoring_tags)}</div>'
        )
        return figure_html.replace("</figcaption>", metrics + "</figcaption>")

    def section(title: str, body: str, is_open: bool = True) -> str:
        flag = " open" if is_open else ""
        return (
            f'<details class="sec"{flag}><summary><h2>{title}</h2></summary>\n'
            f"{body}\n</details>"
        )

    sections = []
    sheets = []
    for key, heading, selection, view_distance in ctx.views():
        if selection is None or view_distance is None:
            continue
        groups = won_against(view_distance, selection)
        picked_here = len(selection.order)
        if key == "combined":
            note = (
                "The pick the tool ships: one distance that mixes tags and image features. "
                "Pick 1 is the most representative image; each later pick adds new coverage. "
                "Hover a pick to see the images it won against, closest first."
            )
            sheets.append('<a href="selected_sheet.png">tags + image</a>')
        else:
            overlap = len(set(selection.order) & combined)
            note = (
                f"{overlap} of these {picked_here} also appear in the tags+image set; "
                "this view ignores the other signal completely."
            )
            sheets.append(f'<a href="selected_{key}_sheet.png">{heading}</a>')
        cards = "".join(
            card(index, selection, key, view_distance, groups) for index in selection.order
        )
        # Only the shipped pick list starts open; the single-signal views are
        # comparison material and stay folded until asked for.
        sections.append(
            section(
                f"Selected - {heading} ({picked_here})",
                f'<p class="note">{note}</p>\n<div class="grid">{cards}</div>',
                is_open=(key == "combined"),
            )
        )

    rest = "".join(
        lost_card(index) for index in range(kept) if index not in combined
    )

    rows = _coverage_rows(ctx)
    coverage = "".join(
        f"<tr><td>{html.escape(tag)}</td><td>{count}</td><td>{picked_count}</td>"
        f"<td class='flag'>{flag}</td></tr>"
        for tag, count, picked_count, flag in rows
    )

    dup_rows = "".join(
        f"<tr><td>{html.escape(record.path.name)}</td><td>{html.escape(record.dup_reason or '')}</td>"
        f"<td>{html.escape(record.duplicate_of.path.name if record.duplicate_of else '')}</td></tr>"
        for record in ctx.records
        if record.duplicate_of is not None
    )

    coverage_section = section(
        "Tag coverage",
        f"""<p class="note">Tags present in the corpus but absent from the selection are where a
feature can dominate training. <code>missing</code>: 3+ source images, none picked.
Computed on the tags+image selection.</p>
<table><thead><tr><th>tag</th><th>images</th><th>picked</th><th></th></tr></thead>
<tbody>{coverage}</tbody></table>""",
    )
    aspect_rows = _aspect_rows(ctx)
    aspect_table = "".join(
        f"<tr><td>{bucket}</td><td>{count}</td><td>{hits}</td><td class='flag'>{flag}</td></tr>"
        for bucket, count, hits, flag in aspect_rows
    )
    corpus_short, picks_short, small = _size_stats(ctx)
    if small:
        shown = ", ".join(
            f"{html.escape(record.path.name)} ({record.width}x{record.height})"
            for record in small[:8]
        )
        more = f" and {len(small) - 8} more" if len(small) > 8 else ""
        small_html = (
            f'<p class="note">{len(small)} images are below {ctx.train_resolution} px on the '
            f"short side and would be upscaled at training time: {shown}{more}</p>"
        )
    else:
        small_html = (
            f'<p class="note">Every image is at least {ctx.train_resolution} px on the short '
            "side, so nothing is upscaled at training time.</p>"
        )
    if ctx.aspect_weight > 0:
        weight_html = (
            f"Tag and image each count 1.00 and frame shape adds up to {ctx.aspect_weight:.2f}; "
            "the combined distance is divided by that total weight at the end, so it stays "
            "0..1. A soft constraint: it bends the picks toward varied framing without "
            "forcing a quota."
        )
    else:
        weight_html = "Frame shape does not affect the pick order (--aspect-weight 0)."
    aspect_section = section(
        "Aspect and size",
        f"""<p class="note">Anima is a DiT trained on aspect buckets, so every image is
resampled before the model sees it and pixel count disappears. Frame shape does not: a
corpus that is mostly one shape teaches that composition along with the style.
{weight_html} Short side of the corpus: {corpus_short[0]} / {corpus_short[1]} /
{corpus_short[2]} px; of the picks: {picks_short[0]} / {picks_short[1]} /
{picks_short[2]} px (min / median / max).</p>
<p class="note"><code>missing</code>: 3+ images of that shape, none picked.
<code>thin</code>: only 1-2 images of that shape, none picked, so the weight has
nothing to balance against. Add more of that shape instead of raising the weight.</p>
<table><thead><tr><th>bucket</th><th>images</th><th>picked</th><th></th></tr></thead>
<tbody>{aspect_table}</tbody></table>
{small_html}""",
    )
    duplicates_section = section(
        "Dropped as duplicates",
        f"""<table><thead><tr><th>file</th><th>reason</th><th>kept instead</th></tr></thead>
<tbody>{dup_rows or '<tr><td colspan="3">none</td></tr>'}</tbody></table>""",
        is_open=False,
    )
    rest_section = section(
        f"Not selected ({kept - picked})",
        f"""<p class="note">Hover a card to see the pick it lost to.</p>
<div class="grid">{rest}</div>""",
    )

    method_label = ctx.selection.method + (" + swap" if ctx.selection.refined else "")
    picks = list(ctx.selection.order)
    if len(picks) > 1:
        sub = ctx.distance[np.ix_(picks, picks)]
        closest_pair = f"{sub[np.triu_indices(len(picks), 1)].min():.3f}"
    else:
        closest_pair = "-"
    template = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>uniqpick - {picked} of {kept}</title>
<style>
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 32px 40px 80px; background: #101012; color: #e8e6e3;
       font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 16px; margin: 40px 0 14px; padding-bottom: 6px; border-bottom: 1px solid #2a2a2e; }}
h2 span {{ color: #8d8d96; font-size: 13px; font-weight: 400; }}
.sub {{ color: #8d8d96; margin-bottom: 22px; }}
.stats {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 18px 0 6px; }}
.stat {{ background: #19191d; border: 1px solid #26262b; border-radius: 8px; padding: 8px 12px; }}
.stat b {{ display: block; font-size: 18px; }}
.stat span {{ color: #8d8d96; font-size: 12px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; }}
.card {{ background: #17171b; border: 1px solid #26262b; border-radius: 10px; overflow: hidden; margin: 0; }}
.card a {{ display: block; color: inherit; text-decoration: none; }}
.card.dim {{ opacity: .55; }}
.card.dim:hover {{ opacity: 1; }}
.shot {{ position: relative; aspect-ratio: 4 / 5; background: #0c0c0e; display: flex; align-items: center; }}
.shot img {{ width: 100%; height: 100%; object-fit: contain; }}
.badge {{ position: absolute; top: 8px; left: 8px; background: #ffd654; color: #1a1a1a;
          font-weight: 700; border-radius: 6px; padding: 1px 7px; font-size: 12px; }}
.shot > a.open {{ display: block; width: 100%; height: 100%; }}
.lost, .won {{ position: absolute; left: 6px; right: 6px; bottom: 6px; background: rgba(8, 8, 10, .94);
         border: 1px solid #2f2f36; border-radius: 6px; padding: 6px; font-size: 11px;
         line-height: 1.4; color: #d6d6dc; opacity: 0; transition: opacity .12s ease;
         pointer-events: auto; text-decoration: none; }}
.lost:hover, .won a:hover {{ border-color: #4a4a54; }}
.lost {{ display: flex; gap: 7px; align-items: flex-start; }}
.card:hover .lost, .card:hover .won, .card:focus-within .lost, .card:focus-within .won {{ opacity: 1; }}
.won {{ display: flex; flex-direction: column; gap: 5px; }}
.wonhead {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.won b, .lost b {{ color: #ffd654; font-weight: 600; }}
.won .dim, .lost .dim {{ color: #9a9aa4; }}
.strip {{ display: flex; gap: 4px; align-items: center; flex-wrap: wrap; }}
.won img {{ width: 38px; height: 51px; object-fit: cover; border-radius: 3px;
            border: 1px solid #3c3c44; flex: none; background: #000; }}
.won .more {{ color: #9a9aa4; font-size: 10px; padding-left: 2px; }}
.lost img {{ width: 62px; height: 82px; object-fit: cover; border-radius: 4px;
             border: 1px solid #3c3c44; flex: none; background: #000; }}
.losttext {{ min-width: 0; }}
.losttext .name {{ font: 10px ui-monospace, monospace; overflow: hidden; text-overflow: ellipsis;
                   white-space: nowrap; }}
.lost b {{ color: #ffd654; font-weight: 600; }}
.lost .dim {{ color: #9a9aa4; }}
figcaption {{ padding: 8px 10px 10px; }}
.fn {{ display: block; font: 11px ui-monospace, monospace; color: #7f7f88; overflow: hidden;
       text-overflow: ellipsis; white-space: nowrap; }}
.fn:hover {{ color: #b9b9c2; }}
.meta {{ font-size: 12px; color: #c9c9d1; margin: 3px 0; }}
.pill {{ display: inline-block; font-size: 10px; line-height: 16px; padding: 0 6px; border-radius: 4px; margin-right: 6px; }}
.pill.shared {{ background: #1d3527; color: #8fe3b0; }}
.pill.solo {{ background: #38281b; color: #ffc98f; }}
.sep {{ color: #3a3a40; }}
.tags {{ font-size: 11px; color: #8d8d96; max-height: 4.6em; overflow: hidden; }}
.more {{ color: #5f5f68; }}
table {{ border-collapse: collapse; width: 100%; max-width: 860px; font-size: 13px; }}
th, td {{ text-align: left; padding: 5px 10px; border-bottom: 1px solid #232327; }}
th {{ color: #8d8d96; font-weight: 500; }}
th:nth-child(2), th:nth-child(3),
td:nth-child(2), td:nth-child(3) {{ text-align: right; }}
.flag {{ color: #ffb84d; }}
code {{ background: #1c1c21; padding: 1px 5px; border-radius: 4px; }}
.note {{ color: #8d8d96; max-width: 860px; }}
a {{ color: #8fc7ff; }}
details.sec {{ margin-top: 34px; }}
details.sec > summary {{ list-style: none; cursor: pointer; user-select: none; }}
details.sec > summary::-webkit-details-marker {{ display: none; }}
details.sec > summary::before {{ content: "\\25B8"; color: #6f6f78; margin-right: 8px;
                                 display: inline-block; transition: transform .12s ease; }}
details.sec[open] > summary::before {{ transform: rotate(90deg); }}
details.sec > summary h2 {{ display: inline-block; margin: 0 0 12px; border-bottom: none; padding: 0; }}
details.sec[open] > summary h2 {{ border-bottom: 1px solid #2a2a2e; padding-bottom: 6px; }}
.foldbar {{ margin: 14px 0 0; }}
.foldbar button {{ font: 12px ui-sans-serif, system-ui, sans-serif; background: #19191d; color: #c9c9d1;
                   border: 1px solid #2a2a2e; border-radius: 6px; padding: 4px 10px; margin-right: 8px;
                   cursor: pointer; }}
.foldbar button:hover {{ border-color: #3c3c44; }}
</style></head><body>
<h1>uniqpick</h1>
<div class="sub">{html.escape(', '.join(str(root) for root in ctx.input_roots))}</div>
<div class="stats">
  <div class="stat"><b>{total}</b><span>images found</span></div>
  <div class="stat"><b>{kept}</b><span>after dedupe</span></div>
  <div class="stat"><b>{picked}</b><span>selected</span></div>
  <div class="stat"><b>{method_label}</b><span>method</span></div>
  <div class="stat"><b>{closest_pair}</b><span>closest pair</span></div>
  <div class="stat"><b>{ctx.tag_weight:.2f} / {ctx.image_weight:.2f} / +{ctx.aspect_weight:.2f}</b><span>tag / image / aspect</span></div>
  <div class="stat"><b>{ctx.backend}</b><span>image features</span></div>
</div>
{short_note}<p class="note"><b>lost to</b> on a dim card means that pick already represents the
image's spot in the corpus: the distance is how far the image sits from it, split into
tag and image parts. Hover a dim card to see the pick itself next to those numbers.
Small means the pick is a near stand-in and the image adds little; large means it was a
real candidate that did not fit the budget. Hover a pick to see the images it
<b>won against</b>, closest first. Sheets: {' &middot; '.join(sheets)}.</p>
<div class="foldbar">
  <button type="button" data-fold="open">Expand all</button>
  <button type="button" data-fold="close">Collapse all</button>
</div>

{''.join(sections)}

{coverage_section}

{aspect_section}

{duplicates_section}

{rest_section}
<script>
document.querySelectorAll('button[data-fold]').forEach(function (button) {{
  button.addEventListener('click', function () {{
    var open = button.dataset.fold === 'open';
    document.querySelectorAll('details.sec').forEach(function (item) {{ item.open = open; }});
  }});
}});
</script>
</body></html>"""
    output.write_text(template, encoding="utf-8")
    return output


def write_json(ctx: ReportContext, names: dict[int, str], output: Path) -> Path:
    def pick_rows(selection: Selection) -> list[dict]:
        return [
            {
                "rank": rank + 1,
                "file": ctx.kept[index].path.name,
                "path": str(ctx.kept[index].path),
                "uniqueness": round(float(ctx.uniqueness[index]), 4),
                "pick_score": round(selection.scores[rank], 4),
                "min_dist_to_previous": round(selection.nearest[rank], 4),
                "in_combined": index in set(ctx.selection.order),
                "aspect_ratio": round(float(ctx.aspect_ratios[index]), 4) if ctx.aspect_ratios is not None else None,
                "aspect_bucket": _aspect_label(ctx, index),
                "tags": sorted(ctx.kept[index].tags),
            }
            for rank, index in enumerate(selection.order)
        ]

    corpus_short, picks_short, small = _size_stats(ctx)
    payload = {
        "method": ctx.selection.method,
        "count": len(ctx.selection.order),
        "requested": ctx.requested or len(ctx.selection.order),
        "tag_weight": ctx.tag_weight,
        "image_weight": ctx.image_weight,
        "aspect_weight": ctx.aspect_weight,
        "train_resolution": ctx.train_resolution,
        "backend": ctx.backend,
        "selected": pick_rows(ctx.selection),
        "selections": {
            key: pick_rows(selection)
            for key, _heading, selection, _distance in ctx.views()
            if selection is not None
        },
        "aspect": {
            "buckets": [
                {"bucket": bucket, "images": count, "picked": hits, "note": flag}
                for bucket, count, hits, flag in _aspect_rows(ctx)
            ],
            "short_side": {"corpus": list(corpus_short), "picks": list(picks_short)},
            "below_training_resolution": len(small),
        },
        "duplicates": [
            {
                "file": record.path.name,
                "kept": record.duplicate_of.path.name if record.duplicate_of else None,
                "reason": record.dup_reason,
            }
            for record in ctx.records
            if record.duplicate_of is not None
        ],
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
