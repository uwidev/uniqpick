"""Tests for uniqpick: caption parsing, dedupe, scoring, selection and the CLI."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from uniqpick import cli, runstore
from uniqpick.captions import META_TAGS, load_tags, parse_tag, split_tags
from uniqpick.data import Record, dedupe, discover
from uniqpick.scoring import (
    aspect_bucket,
    aspect_distance,
    build_tfidf,
    combine,
    scale01,
    tag_distance,
)
from uniqpick.report import won_against
from uniqpick.select import (
    coverage_objective,
    disparity_objective,
    kcenter_objective,
    nearest_selected,
    select,
    select_coverage,
    select_disparity,
    select_kcenter,
)


def make_record(name: str, md5: str, bits: int, tags: dict[str, float] | None = None) -> Record:
    return Record(path=Path(name), width=10, height=10, md5=md5, dhash=bits, tags=tags or {})


class CaptionTests(unittest.TestCase):
    def test_split_ignores_commas_inside_parens(self) -> None:
        parts = split_tags("a, b (x, y), c")
        self.assertEqual(parts, ["a", "b (x, y)", "c"])

    def test_weight_syntax(self) -> None:
        tag = parse_tag("(izumi (idolmaster):1.2)")
        self.assertIsNotNone(tag)
        self.assertEqual(tag.name, "izumi (idolmaster)")
        self.assertAlmostEqual(tag.weight, 1.2)

    def test_brackets_and_underscores(self) -> None:
        self.assertAlmostEqual(parse_tag("[blue_hair]").weight, 0.95)
        self.assertEqual(parse_tag("blue_hair").name, "blue hair")
        self.assertEqual(parse_tag(r"blue\_hair").name, "blue_hair")

    def test_meta_tags_are_dropped(self) -> None:
        self.assertIn("high resolution", META_TAGS)
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "a.jpg"
            image.write_bytes(b"x")
            (Path(folder) / "a.txt").write_text(
                "blue hair, high resolution, @artist", encoding="utf-8"
            )
            (Path(folder) / "a.personal.txt").write_text(
                "rating:sensitive, meta:bad", encoding="utf-8"
            )
            tags = load_tags(image)
            self.assertNotIn("high resolution", tags)
            self.assertNotIn("rating:sensitive", tags)
            self.assertIn("blue hair", tags)
            self.assertIn("@artist", tags)


class DedupeTests(unittest.TestCase):
    def test_exact_and_near_duplicates_collapse(self) -> None:
        records = [
            make_record("a.jpg", "aaa", 0b101010),
            make_record("b.jpg", "aaa", 0b101010),  # exact copy
            make_record("c.jpg", "bbb", 0b101011),  # 1 bit away
            make_record("d.jpg", "ccc", 0b11110000),
        ]
        kept = dedupe(records, max_hamming=2)
        self.assertEqual([r.name for r in kept], ["a.jpg", "d.jpg"])
        dropped = [r for r in records if r.duplicate_of is not None]
        self.assertEqual(len(dropped), 2)
        self.assertEqual(dropped[0].dup_reason, "exact duplicate")
        self.assertEqual(dropped[1].dup_reason, "near duplicate")

    def test_near_duplicates_can_be_disabled(self) -> None:
        records = [
            make_record("a.jpg", "aaa", 0b0000),
            make_record("b.jpg", "bbb", 0b0001),
        ]
        kept = dedupe(records, max_hamming=-1)
        self.assertEqual(len(kept), 2)

    def test_flat_dhash_never_merges(self) -> None:
        records = [make_record("a.jpg", "a", 0), make_record("b.jpg", "b", 0)]
        self.assertEqual(len(dedupe(records, max_hamming=6)), 2)

    def test_transparent_png_matches_white_jpeg(self) -> None:
        # A white-background JPEG and the same art as a transparent PNG differ in
        # md5 and in raw pixel values; they still have to hash as duplicates.
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            art = Image.new("RGB", (120, 120), (250, 250, 250))
            for x in range(20, 100):
                for y in range(30, 90):
                    art.putpixel((x, y), (20 + x, 40, 200 - y))
            art.save(folder / "white.jpg", "JPEG", quality=95)
            rgba = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
            rgba.paste(art, (0, 0))
            rgba.save(folder / "alpha.png")

            records = discover([folder])
            self.assertEqual(len(records), 2)
            self.assertEqual(len(dedupe(records, max_hamming=6)), 1)


class ScoringTests(unittest.TestCase):
    def test_constant_tags_are_ignored(self) -> None:
        docs = [{"style": 1.0, "blue": 1.0}, {"style": 1.0, "red": 1.0}]
        _, vocab, _ = build_tfidf(docs, min_df=1, max_df=0.95)
        self.assertEqual(vocab, ["blue", "red"])

    def test_tag_distance_is_symmetric_with_zero_diagonal(self) -> None:
        docs = [{"a": 1.0}, {"b": 1.0}, {"a": 1.0, "c": 1.0}]
        distance, vocab = tag_distance(docs, min_df=1, max_df=1.0)
        self.assertEqual(len(vocab), 3)
        np.testing.assert_allclose(distance, distance.T, atol=1e-6)
        np.testing.assert_allclose(np.diag(distance), 0.0, atol=1e-6)

    def test_scale01_keeps_gaps_between_close_pairs(self) -> None:
        # A clamped 1st percentile used to flatten every close pair onto 0, which
        # made unrelated images look identical and produced fake selection ties.
        distance = np.array([[0.0, 0.2, 0.3], [0.2, 0.0, 0.5], [0.3, 0.5, 0.0]])
        scaled = scale01(distance)
        self.assertAlmostEqual(float(scaled[0, 1]), 0.0)
        self.assertAlmostEqual(float(scaled[0, 2]), (0.3 - 0.2) / (0.5 - 0.2))
        self.assertAlmostEqual(float(scaled[0, 0]), 0.0)

    def test_combine_drops_the_missing_signal(self) -> None:
        tag = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        combined, tag_weight, image_weight, aspect_weight = combine(tag, None, 1.0, 1.0)
        # Weights are what the caller asked for; the missing side simply drops out.
        self.assertAlmostEqual(tag_weight, 1.0)
        self.assertAlmostEqual(image_weight, 0.0)
        self.assertAlmostEqual(aspect_weight, 0.0)
        np.testing.assert_allclose(combined, tag)

    def test_aspect_distance_uses_log_ratio_and_saturates(self) -> None:
        ratios = np.array([2 / 3, 3 / 4, 1.0, 2.0], dtype=np.float32)
        distance = aspect_distance(ratios)
        np.testing.assert_allclose(np.diag(distance), 0.0, atol=1e-6)
        np.testing.assert_allclose(distance, distance.T, atol=1e-6)
        # 2:3 against 3:4 is a small step, not a bucket flip.
        self.assertAlmostEqual(
            float(distance[0, 1]), abs(float(np.log2((2 / 3) / 0.75))), places=5
        )
        # A doubling is distance 1; everything past that clips.
        self.assertAlmostEqual(float(distance[2, 3]), 1.0, places=5)
        self.assertEqual(aspect_bucket(2 / 3), "tall")
        self.assertEqual(aspect_bucket(1.0), "square")
        self.assertEqual(aspect_bucket(16 / 9), "wide")

    def test_combine_soft_aspect_weight_adds_signal(self) -> None:
        tag = np.zeros((2, 2), dtype=np.float32)
        aspect = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        combined, tag_weight, image_weight, aspect_weight = combine(
            tag, None, 1.0, 0.0, aspect_dist=aspect, aspect_weight=0.3
        )
        # Aspect is added on top, so tag and image keep their own weights.
        self.assertAlmostEqual(tag_weight, 1.0)
        self.assertAlmostEqual(image_weight, 0.0)
        self.assertAlmostEqual(aspect_weight, 0.3)
        # 0.3 * aspect is the raw sum; the one final division by 1.3 scales it.
        self.assertAlmostEqual(float(combined[0, 1]), 0.3 / 1.3, places=6)
        self.assertAlmostEqual(float(combined[1, 1]), 0.0, places=6)

    def test_combine_keeps_tag_image_balance_with_aspect(self) -> None:
        tag = np.full((3, 3), 0.5, dtype=np.float32)
        image = np.full((3, 3), 0.5, dtype=np.float32)
        aspect = np.full((3, 3), 1.0, dtype=np.float32)
        combined, tag_weight, image_weight, _ = combine(
            tag, image, 1.0, 1.0, aspect_dist=aspect, aspect_weight=0.3
        )
        # Weights stay exactly as configured; nothing is renormalized midway.
        self.assertAlmostEqual(tag_weight, 1.0)
        self.assertAlmostEqual(image_weight, 1.0)
        # raw = 1.0 * 0.5 + 1.0 * 0.5 + 0.3 * 1.0 = 1.3, divided by 2.3 at the end.
        self.assertAlmostEqual(float(combined.max()), 1.3 / 2.3, places=6)
        # Every signal at its maximum still lands on 1.0 after the final division.
        top, _, _, _ = combine(
            np.ones((3, 3), dtype=np.float32),
            np.ones((3, 3), dtype=np.float32),
            1.0,
            1.0,
            aspect_dist=np.ones((3, 3), dtype=np.float32),
            aspect_weight=0.3,
        )
        self.assertAlmostEqual(float(top.max()), 1.0, places=6)


def _random_distance(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3))
    gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    return (gaps / gaps.max()).astype(np.float32)


class SelectionTests(unittest.TestCase):
    def test_coverage_picks_distinct_entries(self) -> None:
        selection = select_coverage(np.eye(4, dtype=np.float32), 3)
        self.assertEqual(len(set(selection.order)), 3)
        self.assertEqual(selection.order[0], 0)

    def test_kcenter_starts_at_medoid_then_far_point(self) -> None:
        distance = np.array(
            [
                [0.0, 0.1, 0.9, 0.9],
                [0.1, 0.0, 0.9, 0.9],
                [0.9, 0.9, 0.0, 0.2],
                [0.9, 0.9, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        selection = select_kcenter(distance, 2)
        self.assertEqual(selection.order[0], 0)
        self.assertEqual(selection.order[1], 2)

    def test_refinement_never_lowers_coverage(self) -> None:
        for seed in range(5):
            distance = _random_distance(40, seed)
            similarity = 1.0 - distance
            raw = select_coverage(similarity, 6, refine=False)
            refined = select_coverage(similarity, 6, refine=True)
            self.assertGreaterEqual(
                coverage_objective(similarity, refined.order),
                coverage_objective(similarity, raw.order) - 1e-6,
            )
            self.assertEqual(len(set(refined.order)), 6)

    def test_refinement_is_stable(self) -> None:
        distance = _random_distance(30, 7)
        similarity = 1.0 - distance
        first = select_coverage(similarity, 5, refine=True)
        second = select_coverage(similarity, 5, refine=True)
        self.assertEqual(first.order, second.order)

    def test_kcenter_refinement_raises_the_weakest_link(self) -> None:
        line = np.abs(np.subtract.outer(np.arange(4.0), np.arange(4.0))).astype(np.float32)
        # Greedy starts at a medoid then takes the far end: {1, 3}, gap 2.
        raw = select_kcenter(line, 2, refine=False)
        self.assertEqual(set(raw.order), {1, 3})
        refined = select_kcenter(line, 2, refine=True)
        self.assertEqual(set(refined.order), {0, 3})
        self.assertAlmostEqual(kcenter_objective(line, refined.order), 3.0)

    def test_disparity_picks_the_two_extremes(self) -> None:
        line = np.abs(np.subtract.outer(np.arange(4.0), np.arange(4.0))).astype(np.float32)
        selection = select_disparity(line, 2)
        self.assertEqual(set(selection.order), {0, 3})
        self.assertAlmostEqual(disparity_objective(line, selection.order), 3.0)

    def test_replayed_scores_add_up_to_the_objective(self) -> None:
        distance = _random_distance(25, 3)
        similarity = 1.0 - distance
        coverage = select_coverage(similarity, 4)
        self.assertAlmostEqual(sum(coverage.scores), coverage.objective, places=5)
        spread = select_disparity(distance, 4)
        self.assertAlmostEqual(sum(spread.scores), spread.objective, places=5)
        spread_k = select_kcenter(distance, 4)
        # Every unordered pair shows up once as a later pick's nearest previous one.
        self.assertAlmostEqual(min(spread_k.scores[1:]), spread_k.objective, places=5)

    def test_unknown_method_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select(np.eye(3, dtype=np.float32), 2, method="nope")

    def test_nearest_selected_reports_closest_pick(self) -> None:
        distance = np.array(
            [
                [0.0, 0.1, 0.9, 0.8],
                [0.1, 0.0, 0.7, 0.6],
                [0.9, 0.7, 0.0, 0.2],
                [0.8, 0.6, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        nearest, gaps = nearest_selected(distance, [0, 2])
        np.testing.assert_array_equal(nearest, [0, 0, 2, 2])
        np.testing.assert_allclose(gaps, [0.0, 0.1, 0.0, 0.2], atol=1e-6)


def write_image(path: Path, seed: int) -> None:
    """Textured noise: flat colours share a dhash, so they cannot test any of this."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(array).save(path, "JPEG", quality=95)


class ReportTests(unittest.TestCase):
    def test_won_against_partitions_the_corpus(self) -> None:
        # Every image that is not a pick lost to exactly one pick.
        distance = np.array(
            [
                [0.0, 0.9, 0.2, 0.7, 0.3],
                [0.9, 0.0, 0.8, 0.1, 0.85],
                [0.2, 0.8, 0.0, 0.6, 0.1],
                [0.7, 0.1, 0.6, 0.0, 0.75],
                [0.3, 0.85, 0.1, 0.75, 0.0],
            ]
        )
        selection = select_coverage(1.0 - distance, 2)
        groups = won_against(distance, selection)
        owned = [row for rows in groups.values() for row, _gap in rows]
        expected = [i for i in range(5) if i not in set(selection.order)]
        self.assertEqual(sorted(owned), sorted(expected))
        self.assertEqual(len(owned), len(set(owned)))
        for rows in groups.values():
            gaps = [gap for _row, gap in rows]
            self.assertEqual(gaps, sorted(gaps))

    def test_aspect_rows_flags_missing_and_thin_shapes(self) -> None:
        from types import SimpleNamespace

        from uniqpick.report import _aspect_rows

        # 3 tall (none picked), 2 square (one picked), 1 wide (none picked).
        ratios = np.array([0.7, 0.7, 0.7, 1.0, 1.0, 1.6], dtype=np.float32)
        ctx = SimpleNamespace(
            kept=[Record(path=Path(f"{i}.jpg"), width=10, height=10, md5="", dhash=0) for i in range(6)],
            selection=SimpleNamespace(order=[3]),
            aspect_ratios=ratios,
        )
        rows = {bucket: (count, hits, flag) for bucket, count, hits, flag in _aspect_rows(ctx)}
        self.assertEqual(rows["tall"], (3, 0, "missing"))
        self.assertEqual(rows["square"], (2, 1, ""))
        self.assertEqual(rows["wide"], (1, 0, "thin"))


class CliTests(unittest.TestCase):
    def test_end_to_end_on_synthetic_folder(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "imgs"
            source.mkdir()
            items = [
                ("a.jpg", "red, solo"),
                ("b.jpg", "green, duo"),
                ("c.jpg", "blue, solo"),
                ("d.jpg", "yellow, crowd"),
                ("e.jpg", "cyan, solo"),
                ("f.jpg", "magenta, duo"),
            ]
            for seed, (name, caption) in enumerate(items):
                write_image(source / name, seed)
                (source / (Path(name).stem + ".txt")).write_text(caption, encoding="utf-8")

            out = Path(folder) / "out"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                code = cli.main(
                    [str(source), "-k", "2", "--embeddings", "none", "-o", str(out), "--quiet"]
                )
            self.assertEqual(code, 0, stderr.getvalue())

            self.assertEqual(len(list((out / "selected").glob("*.jpg"))), 2)
            for image in (out / "selected").glob("*.jpg"):
                self.assertTrue(image.with_suffix(".txt").exists(), f"missing caption for {image.name}")
            self.assertTrue((out / "selection.csv").exists())
            self.assertTrue((out / "report.html").exists())
            self.assertTrue((out / "selection.json").exists())
            self.assertTrue((out / "report.md").exists())
            self.assertTrue((out / "selected_sheet.png").exists())
            page = (out / "report.html").read_text(encoding="utf-8")
            self.assertIn('class="won"', page)
            self.assertIn("won against", page)
            # Overlay thumbnails are links to their own file, not the card's image.
            self.assertIn('class="open"', page)
            self.assertIn('class="lost"', page)
            self.assertIn('<details class="sec"', page)
            self.assertIn('data-fold="close"', page)
            # The shipped pick list opens; the single-signal comparison stays folded.
            self.assertIn('<details class="sec" open><summary><h2>Selected - tags + image', page)
            self.assertIn('<details class="sec"><summary><h2>Selected - tags only', page)
            self.assertIn("Aspect and size", page)
            header = (out / "selection.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("aspect_ratio", header)
            self.assertIn("aspect_distance", header)

    def test_count_is_clamped_to_corpus_size(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "imgs"
            source.mkdir()
            write_image(source / "only.jpg", 1)
            (source / "only.txt").write_text("solo", encoding="utf-8")
            out = Path(folder) / "out"
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                code = cli.main([str(source), "-k", "5", "--embeddings", "none", "-o", str(out), "--quiet"])
            self.assertEqual(code, 0)
            self.assertEqual(len(list((out / "selected").glob("*.jpg"))), 1)
            payload = json.loads((out / "selection.json").read_text(encoding="utf-8"))
            # The report keeps the request and the actual count apart.
            self.assertEqual(payload["requested"], 5)
            self.assertEqual(payload["count"], 1)
            html = (out / "report.html").read_text(encoding="utf-8")
            self.assertIn("5</b> picks requested", html)


class TwoStageTests(unittest.TestCase):
    """compute once, then pick repeatedly from the stored distances."""

    ITEMS = [
        ("a.jpg", "red, solo"),
        ("b.jpg", "green, duo"),
        ("c.jpg", "blue, solo"),
        ("d.jpg", "yellow, crowd"),
        ("e.jpg", "cyan, solo"),
        ("f.jpg", "magenta, duo"),
    ]

    def make_corpus(self, folder: Path) -> Path:
        source = folder / "imgs"
        source.mkdir()
        for seed, (name, caption) in enumerate(self.ITEMS):
            write_image(source / name, seed)
            (source / (Path(name).stem + ".txt")).write_text(caption, encoding="utf-8")
        return source

    def call(self, argv: list[str]) -> tuple[int, str]:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            code = cli.main(argv)
        return code, stderr.getvalue()

    def compute(self, source: Path, run: Path, *extra: str) -> None:
        code, stderr = self.call(
            ["compute", str(source), "-o", str(run), "--embeddings", "none", "--quiet", *extra]
        )
        self.assertEqual(code, 0, stderr)

    def test_compute_then_pick_reuses_stored_distances(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = self.make_corpus(Path(folder))
            run = Path(folder) / "run"

            self.compute(source, run)
            self.assertTrue((run / "meta.json").is_file())
            self.assertTrue((run / "distances.npz").is_file())
            # Computing does not pick; the run folder stays free of report output.
            self.assertFalse((run / "selected").exists())
            self.assertFalse((run / "report.html").exists())

            out2 = Path(folder) / "k2"
            out4 = Path(folder) / "k4"
            code, stderr = self.call(["pick", str(run), "-k", "2", "-o", str(out2), "--quiet"])
            self.assertEqual(code, 0, stderr)
            code, stderr = self.call(["pick", str(run), "-k", "4", "-o", str(out4), "--quiet"])
            self.assertEqual(code, 0, stderr)

            names2 = {path.name for path in (out2 / "selected").glob("*.jpg")}
            self.assertEqual(len(names2), 2)
            self.assertEqual(len(list((out4 / "selected").glob("*.jpg"))), 4)
            self.assertTrue((out2 / "report.html").is_file())
            # Coverage is a prefix walk, so k=2 is the first two picks of k=4.
            payload = json.loads((out4 / "selection.json").read_text(encoding="utf-8"))
            self.assertEqual(names2, {row["file"] for row in payload["selected"][:2]})

    def test_pick_supports_disparity_and_no_refine(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = self.make_corpus(Path(folder))
            run = Path(folder) / "run"
            self.compute(source, run)
            out = Path(folder) / "spread"
            code, stderr = self.call(
                ["pick", str(run), "-k", "3", "--method", "disparity", "--no-refine",
                 "-o", str(out), "--quiet"]
            )
            self.assertEqual(code, 0, stderr)
            page = (out / "report.html").read_text(encoding="utf-8")
            self.assertIn("disparity", page)
            self.assertNotIn("+ swap", page)

    def test_pick_defaults_to_a_k_folder_next_to_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = self.make_corpus(Path(folder))
            run = Path(folder) / "run"
            self.compute(source, run)
            code, stderr = self.call(["pick", str(run), "-k", "3", "--quiet"])
            self.assertEqual(code, 0, stderr)
            self.assertTrue((run / "k3" / "report.html").is_file())
            self.assertTrue((run / "k3" / "selection.csv").is_file())

    def test_one_shot_run_can_be_picked_again(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = self.make_corpus(Path(folder))
            out = Path(folder) / "out"
            code, stderr = self.call(
                [str(source), "-k", "2", "--embeddings", "none", "-o", str(out), "--quiet"]
            )
            self.assertEqual(code, 0, stderr)
            self.assertTrue((out / "meta.json").is_file())

            again = Path(folder) / "again"
            code, stderr = self.call(["pick", str(out), "-k", "5", "-o", str(again), "--quiet"])
            self.assertEqual(code, 0, stderr)
            self.assertEqual(len(list((again / "selected").glob("*.jpg"))), 5)

    def test_pick_overrides_weights_without_recomputing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = self.make_corpus(Path(folder))
            run = Path(folder) / "run"
            self.compute(source, run, "--embeddings", "basic")

            flat = Path(folder) / "flat"
            shaped = Path(folder) / "shaped"
            code, stderr = self.call(
                ["pick", str(run), "-k", "3", "-o", str(flat), "--aspect-weight", "0", "--quiet"]
            )
            self.assertEqual(code, 0, stderr)
            code, stderr = self.call(
                ["pick", str(run), "-k", "3", "-o", str(shaped), "--aspect-weight", "1.0", "--quiet"]
            )
            self.assertEqual(code, 0, stderr)

            flat_payload = json.loads((flat / "selection.json").read_text(encoding="utf-8"))
            shaped_payload = json.loads((shaped / "selection.json").read_text(encoding="utf-8"))
            self.assertEqual(flat_payload["aspect_weight"], 0.0)
            self.assertEqual(shaped_payload["aspect_weight"], 1.0)
            # The tag/image base balance is untouched by the aspect override.
            self.assertEqual(flat_payload["tag_weight"], 1.0)
            self.assertEqual(flat_payload["image_weight"], 1.0)

    def test_pick_can_drop_a_tag_and_rebuild_the_tag_distance(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = self.make_corpus(Path(folder))
            run = Path(folder) / "run"
            self.compute(source, run)
            out = Path(folder) / "no-solo"
            code, stderr = self.call(
                ["pick", str(run), "-k", "2", "-o", str(out), "--exclude-tag", "solo", "--quiet"]
            )
            self.assertEqual(code, 0, stderr)
            self.assertEqual(len(list((out / "selected").glob("*.jpg"))), 2)

    def test_pick_rejects_a_folder_that_is_not_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            code, stderr = self.call(["pick", folder, "-k", "2", "--quiet"])
            self.assertEqual(code, 1)
            self.assertIn("not a uniqpick run", stderr.replace("\n", " "))


class RunStoreTests(unittest.TestCase):
    def test_records_round_trip_with_duplicate_links(self) -> None:
        first = make_record("a.jpg", "m1", 1, {"red": 1.0, "solo": 0.5})
        second = make_record("b.jpg", "m2", 2, {"red": 1.0})
        second.duplicate_of = first
        second.dup_reason = "near duplicate"

        back = runstore.decode_records(runstore.encode_records([first, second]))
        self.assertEqual(back[0].tags, {"red": 1.0, "solo": 0.5})
        self.assertIs(back[1].duplicate_of, back[0])
        self.assertEqual(back[1].dup_reason, "near duplicate")

    def test_saved_matrices_reload_and_bad_ones_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            records = [make_record("a.jpg", "m1", 1), make_record("b.jpg", "m2", 2)]
            meta = {
                "version": runstore.RUN_VERSION,
                "records": runstore.encode_records(records),
                "kept": [0, 1],
                "tag_vocab": [],
                "tag_freq": {},
            }
            tag = np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float32)
            runstore.save_run(run, meta, tag, None, None)
            loaded = runstore.load_run(run)
            self.assertEqual(len(loaded.kept), 2)
            np.testing.assert_allclose(loaded.tag_dist, tag)
            self.assertIsNone(loaded.image_dist)

            np.savez(run / runstore.DIST_NAME, tag=np.ones((2, 2), dtype=np.float32))
            with self.assertRaises(runstore.RunFormatError):
                runstore.load_run(run)

    def test_version_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            run.mkdir()
            (run / runstore.META_NAME).write_text(
                json.dumps({"version": 999, "records": [], "kept": []}), encoding="utf-8"
            )
            with self.assertRaises(runstore.RunFormatError):
                runstore.load_run(run)



if __name__ == "__main__":
    unittest.main()
