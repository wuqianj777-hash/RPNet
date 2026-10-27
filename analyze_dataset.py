from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from rpnet.ccpd import decode_plate, parse_ccpd_filename


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize CCPD2020 annotations")
    parser.add_argument("--data-root", default="CCPD2020/ccpd_green")
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()
    root, output = Path(args.data_root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    all_brightness, all_blur = [], []
    for split in ("train", "val", "test"):
        paths = sorted((root / split).glob("*.jpg"))
        annotations = [parse_ccpd_filename(path) for path in paths]
        all_brightness.extend(item.brightness for item in annotations)
        all_blur.extend(item.blurriness for item in annotations)
        report[split] = {
            "images": len(paths),
            "brightness_mean": float(np.mean([item.brightness for item in annotations])),
            "blurriness_mean": float(np.mean([item.blurriness for item in annotations])),
            "province_counts": dict(Counter(item.plate[0] for item in annotations)),
        }
    report["total"] = {"images": sum(value["images"] for value in report.values())}
    (output / "dataset_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(all_brightness, bins=30, color="#287271")
    axes[0].set(title="Brightness distribution", xlabel="Brightness", ylabel="Images")
    axes[1].hist(all_blur, bins=30, color="#d1662f")
    axes[1].set(title="Blurriness distribution", xlabel="Score", ylabel="Images")
    fig.tight_layout()
    fig.savefig(output / "attribute_distributions.png", dpi=180)
    plt.close(fig)

    samples = sorted((root / "val").glob("*.jpg"))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    for axis, path in zip(axes.flat, samples):
        image = cv2.imread(str(path))
        ann = parse_ccpd_filename(path)
        x1, y1, x2, y2 = ann.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(decode_plate(ann.plate), fontfamily="SimHei")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output / "annotation_samples.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

