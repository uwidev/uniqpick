"""Image feature extraction: perceptual hashes and dense embeddings."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


# --------------------------------------------------------------------------- #
# Hashing / basic image facts
# --------------------------------------------------------------------------- #


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path, size: int = 8) -> int:
    """Difference hash: 64 bits describing the horizontal gradient layout.

    Goes through ``open_rgb`` so a transparent PNG and a white-background JPEG
    of the same picture hash the same; hashing the raw pixels would put the
    transparent area at a different luminance and miss the duplicate.
    """
    gray = open_rgb(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def open_rgb(path: Path, max_side: int | None = None) -> Image.Image:
    """Open an image as RGB, honouring EXIF orientation and removing alpha."""
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("RGB", "L"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        transparent = image.mode in ("RGBA", "LA", "PA") or (
            image.mode == "P" and "transparency" in image.info
        )
        if transparent:
            rgba = image.convert("RGBA")
            background.paste(rgba, mask=rgba.getchannel("A"))
        else:
            background.paste(image.convert("RGB"))
        image = background
    else:
        image = image.convert("RGB")
    if max_side is not None and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    return image


# --------------------------------------------------------------------------- #
# Embedding backends
# --------------------------------------------------------------------------- #


class EmbeddingBackend:
    """Dense image features.  Subclasses fill ``embed``."""

    name = "base"
    #: Whether per-dimension z-scoring helps before comparing vectors.
    standardize = False

    def embed(self, paths: list[Path], progress=None) -> np.ndarray:
        raise NotImplementedError


class ClipBackend(EmbeddingBackend):
    """SigLIP / CLIP image embeddings through ``open_clip`` (CPU or CUDA)."""

    name = "clip"
    standardize = False

    def __init__(
        self,
        model: str = "hf-hub:timm/ViT-B-16-SigLIP2",
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        import open_clip  # imported lazily so the basic backend needs no torch

        self._torch = __import__("torch")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model, device=device
        )
        self._model.eval()
        self._device = device
        self._batch_size = batch_size

    def embed(self, paths: list[Path], progress=None) -> np.ndarray:
        torch = self._torch
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(paths), self._batch_size):
                batch = paths[start : start + self._batch_size]
                tensors = torch.stack(
                    [self._preprocess(open_rgb(p, max_side=1024)) for p in batch]
                ).to(self._device)
                features = self._model.encode_image(tensors)
                features = torch.nn.functional.normalize(features, dim=-1)
                chunks.append(features.cpu().numpy())
                if progress:
                    progress(min(start + len(batch), len(paths)), len(paths))
        return np.concatenate(chunks, axis=0)


class BasicBackend(EmbeddingBackend):
    """Dependency-light visual features: colour, layout and edge energy.

    Good enough to separate palettes and compositions when CLIP is not
    available; much weaker than a real embedding at telling subjects apart.
    """

    name = "basic"
    standardize = True

    def embed(self, paths: list[Path], progress=None) -> np.ndarray:
        rows = [self._one(path) for path in paths]
        if progress:
            progress(len(paths), len(paths))
        return np.stack(rows)

    @staticmethod
    def _one(path: Path) -> np.ndarray:
        image = open_rgb(path, max_side=256).resize((256, 256), Image.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

        # 8x8 layout of luminance and saturation.
        lum_grid = lum.reshape(8, 32, 8, 32).mean(axis=(1, 3)).flatten()
        maxc = rgb.max(axis=2)
        minc = rgb.min(axis=2)
        sat = np.where(maxc > 0, (maxc - minc) / np.maximum(maxc, 1e-6), 0.0)
        sat_grid = sat.reshape(4, 64, 4, 64).mean(axis=(1, 3)).flatten()

        # Coarse hue histogram, weighted by saturation so grey art does not
        # produce a rainbow.
        hue = _hue(rgb)
        bins = np.minimum((hue * 12).astype(np.int32), 11)
        hist = np.bincount(bins[sat > 0.15], weights=sat[sat > 0.15], minlength=12)
        total = hist.sum()
        hist = hist / total if total > 1e-6 else hist

        edge = np.abs(np.diff(lum, axis=1)).mean(axis=0)
        edge = np.pad(edge, (0, 256 - edge.size), mode="edge")
        edge_grid = edge.reshape(4, 64).mean(axis=1)

        scalars = np.array(
            [
                lum.mean(),
                lum.std(),
                sat.mean(),
                sat.std(),
                float((np.abs(np.diff(lum, axis=0)) > 0.08).mean()),
                float(image.width / image.height),
            ],
            dtype=np.float32,
        )
        return np.concatenate([lum_grid, sat_grid, hist, edge_grid, scalars]).astype(
            np.float32
        )


def _hue(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB -> hue in [0, 1)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    delta = maxc - minc
    hue = np.zeros_like(maxc)
    mask = delta > 1e-6
    safe = np.where(mask, delta, 1.0)
    rc = np.where(mask & (maxc == r), ((g - b) / safe) % 6.0, 0.0)
    gc = np.where(mask & (maxc == g) & (maxc != r), (b - r) / safe + 2.0, 0.0)
    bc = np.where(mask & (maxc == b) & (maxc != r) & (maxc != g), (r - g) / safe + 4.0, 0.0)
    hue = (rc + gc + bc) / 6.0
    return np.mod(hue, 1.0)


def make_backend(name: str, model: str, device: str) -> EmbeddingBackend:
    if name == "clip":
        return ClipBackend(model=model, device=device)
    if name == "basic":
        return BasicBackend()
    raise ValueError(f"unknown embedding backend: {name}")
