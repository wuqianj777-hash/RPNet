from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


PROVINCES = list("皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新")
LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
# CCPD uses the 24 letters first, followed by digits. Day5's worked example
# 3_24_28_24_31_33 -> D04079 confirms this order.
ALPHANUMERIC = LETTERS + list("0123456789")
CLASS_COUNTS = [len(PROVINCES), len(LETTERS)] + [len(ALPHANUMERIC)] * 6


@dataclass(frozen=True)
class CCPDAnnotation:
    area: float
    tilt: tuple[int, int]
    bbox: tuple[int, int, int, int]
    vertices: tuple[tuple[int, int], ...]
    plate: tuple[int, ...]
    brightness: int
    blurriness: int


def _point(value: str) -> tuple[int, int]:
    x, y = value.split("&")
    return int(x), int(y)


def parse_ccpd_filename(path: str | Path) -> CCPDAnnotation:
    """Parse the seven CCPD fields encoded in an image filename."""
    fields = Path(path).stem.split("-")
    if len(fields) != 7:
        raise ValueError(f"Expected 7 CCPD fields, got {len(fields)}: {path}")
    tilt = tuple(int(v) for v in fields[1].split("_"))
    top_left, bottom_right = (_point(v) for v in fields[2].split("_"))
    vertices = tuple(_point(v) for v in fields[3].split("_"))
    plate = tuple(int(v) for v in fields[4].split("_"))
    if len(plate) != 8:
        raise ValueError(f"CCPD2020 requires 8 plate symbols: {path}")
    x1, y1 = top_left
    x2, y2 = bottom_right
    return CCPDAnnotation(
        area=float(fields[0]),
        tilt=(tilt[0], tilt[1]),
        bbox=(x1, y1, x2, y2),
        vertices=vertices,
        plate=plate,
        brightness=int(fields[5]),
        blurriness=int(fields[6]),
    )


def decode_plate(indices: list[int] | tuple[int, ...]) -> str:
    if len(indices) != 8:
        raise ValueError("A CCPD2020 plate must contain 8 symbols")
    tables = [PROVINCES, LETTERS] + [ALPHANUMERIC] * 6
    return "".join(table[index] for table, index in zip(tables, indices))


def apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def identity(image: np.ndarray) -> np.ndarray:
    return image


PREPROCESSORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "none": identity,
    "clahe": apply_clahe,
}


class CCPD2020Dataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: tuple[int, int] = (256, 416),
        preprocess: str = "none",
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.paths = sorted((self.root / split).glob("*.jpg"))
        if not self.paths:
            raise FileNotFoundError(f"No JPG images found in {self.root / split}")
        if preprocess not in PREPROCESSORS:
            raise ValueError(f"Unknown preprocess '{preprocess}'")
        self.height, self.width = image_size
        self.preprocess = PREPROCESSORS[preprocess]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.paths[index]
        annotation = parse_ccpd_filename(path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        image = self.preprocess(image)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if self.augment:
            alpha = float(np.random.uniform(0.85, 1.15))
            beta = float(np.random.uniform(-12.0, 12.0))
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        x1, y1, x2, y2 = annotation.bbox
        box = torch.tensor(
            [x1 / original_w, y1 / original_h, x2 / original_w, y2 / original_h],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)
        return {
            "image": tensor,
            "box": box,
            "plate": torch.tensor(annotation.plate, dtype=torch.long),
            "path": str(path),
            "brightness": annotation.brightness,
            "blurriness": annotation.blurriness,
        }
