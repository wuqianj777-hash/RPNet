from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from rpnet.ccpd import PREPROCESSORS, decode_plate
from rpnet.metrics import logits_to_plate
from rpnet.model import RPNet8


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCPD2020 plate detection and recognition on one image")
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/clahe_finetuned_best.pt")
    parser.add_argument("--config", default="configs/ccpd2020.yaml")
    parser.add_argument("--preprocess", choices=("none", "clahe"), default="clahe")
    parser.add_argument("--output", default="outputs/demo_prediction.jpg")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RPNet8(tuple(config["roi_size"])).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Could not read {args.image}")
    rgb_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb_image.shape[:2]
    processed = PREPROCESSORS[args.preprocess](rgb_image.copy())
    resized = cv2.resize(processed, (config["image_size"][1], config["image_size"][0]), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float().div(255).unsqueeze(0).to(device)
    with torch.inference_mode():
        box, logits, _ = model(tensor)
        indices = logits_to_plate(logits)[0].tolist()
        confidence = float(torch.stack([value.softmax(1).max(1).values for value in logits], 1).mean())
    x1, y1, x2, y2 = box[0].cpu().numpy()
    points = (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))
    plate = decode_plate(indices)

    canvas = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(points, outline=(220, 55, 45), width=5)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 34) if font_path.exists() else ImageFont.load_default()
    label = f"{plate}  conf={confidence:.3f}"
    text_box = draw.textbbox((points[0], max(0, points[1] - 44)), label, font=font)
    draw.rectangle(text_box, fill=(220, 55, 45))
    draw.text((text_box[0], text_box[1]), label, font=font, fill="white")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=95)
    print(f"plate={plate} confidence={confidence:.4f} box={points} output={destination}")


if __name__ == "__main__":
    main()

