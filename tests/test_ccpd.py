from pathlib import Path

import torch

from rpnet.ccpd import CCPD2020Dataset, decode_plate, parse_ccpd_filename
from rpnet.model import RPNet8


DATA_ROOT = Path("CCPD2020/ccpd_green")


def test_parse_real_filename() -> None:
    path = next((DATA_ROOT / "train").glob("*.jpg"))
    annotation = parse_ccpd_filename(path)
    assert len(annotation.plate) == 8
    assert len(annotation.vertices) == 4
    assert len(decode_plate(annotation.plate)) == 8


def test_decode_day5_green_plate_example() -> None:
    assert decode_plate((0, 0, 3, 24, 28, 24, 31, 33)) == "皖AD04079"


def test_dataset_and_model_shapes() -> None:
    dataset = CCPD2020Dataset(DATA_ROOT, "val")
    sample = dataset[0]
    model = RPNet8()
    boxes, logits, _ = model(sample["image"].unsqueeze(0), sample["box"].unsqueeze(0))
    assert boxes.shape == (1, 4)
    assert len(logits) == 8
    assert [value.shape[1] for value in logits] == [31, 24, 34, 34, 34, 34, 34, 34]
    assert torch.all((boxes >= 0) & (boxes <= 1))
