import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = Path("models/yolo11n-v3/best.pt")
IMAGE_DIR = Path("testing_dataset/License-Plate-Data/test/images")
LABEL_DIR = Path("testing_dataset/License-Plate-Data/test/labels")

CONF = 0.38
IOU_MATCH = 0.50
INFERENCE_SIZE = 640


def yolo_to_xyxy(line, width, height):
    parts = line.split()

    if len(parts) != 5:
        return None

    cls, xc, yc, w, h = map(float, parts)

    x1 = (xc - w / 2) * width
    y1 = (yc - h / 2) * height
    x2 = (xc + w / 2) * width
    y2 = (yc + h / 2) * height

    return {
        "class_id": int(cls),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def box_iou(a, b):
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_ground_truth(label_path, width, height):
    if not label_path.exists():
        return []

    boxes = []

    for line in label_path.read_text().splitlines():
        line = line.strip()

        if not line:
            continue

        box = yolo_to_xyxy(line, width, height)

        if box:
            boxes.append(box)

    return boxes


def main(output_dir: Path = Path("evaluation/results")) -> None:
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Model weights not found: {MODEL_PATH}")
    if not IMAGE_DIR.is_dir() or not LABEL_DIR.is_dir():
        raise SystemExit(
            "Evaluation dataset is unavailable. Expected image and label folders at "
            f"{IMAGE_DIR} and {LABEL_DIR}."
        )

    images = sorted(
        p
        for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        raise SystemExit(f"No supported evaluation images found in {IMAGE_DIR}.")
    missing_labels = [
        path.stem for path in images if not (LABEL_DIR / f"{path.stem}.txt").is_file()
    ]
    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        raise SystemExit(
            f"Missing labels for {len(missing_labels)} evaluation image(s): {preview}"
        )

    print("=" * 70)
    print("YOLO11n V3 HOLDOUT EVALUATION")
    print("=" * 70)
    print(f"Model:          {MODEL_PATH}")
    print(f"Images:         {IMAGE_DIR}")
    print(f"Labels:         {LABEL_DIR}")
    print(f"Test images:    {len(images)}")
    print(f"Confidence:     {CONF}")
    print(f"IoU threshold:  {IOU_MATCH}")
    print("=" * 70)

    model = YOLO(str(MODEL_PATH))

    tp = 0
    fp = 0
    fn = 0

    total_gt = 0
    total_predictions = 0

    inference_times = []
    confidences = []

    images_with_no_detection = 0
    images_with_multiple_predictions = 0

    per_image = []

    for index, image_path in enumerate(images, start=1):

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        # Get image dimensions through model prediction result.
        start = time.perf_counter()

        results = model.predict(
            source=str(image_path),
            conf=CONF,
            iou=0.5,
            imgsz=INFERENCE_SIZE,
            verbose=False,
        )

        elapsed = time.perf_counter() - start
        inference_times.append(elapsed)

        result = results[0]

        height, width = result.orig_shape

        ground_truth = load_ground_truth(
            label_path,
            width,
            height,
        )

        predictions = []

        if result.boxes is not None:
            for box, confidence, cls in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                strict=True,
            ):
                predictions.append(
                    {
                        "class_id": int(cls),
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3],
                        "confidence": float(confidence),
                    }
                )

                confidences.append(float(confidence))

        total_gt += len(ground_truth)
        total_predictions += len(predictions)

        if not predictions:
            images_with_no_detection += 1

        if len(predictions) > 1:
            images_with_multiple_predictions += 1

        # Highest-confidence predictions matched first.
        predictions.sort(
            key=lambda x: x["confidence"],
            reverse=True,
        )

        matched_gt = set()

        image_tp = 0
        image_fp = 0

        for prediction in predictions:

            best_iou = 0.0
            best_gt = None

            for gt_index, gt in enumerate(ground_truth):

                if gt_index in matched_gt:
                    continue

                if prediction["class_id"] != gt["class_id"]:
                    continue

                current_iou = box_iou(prediction, gt)

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt = gt_index

            if best_gt is not None and best_iou >= IOU_MATCH:
                matched_gt.add(best_gt)
                tp += 1
                image_tp += 1
            else:
                fp += 1
                image_fp += 1

        image_fn = len(ground_truth) - len(matched_gt)

        fn += image_fn

        per_image.append(
            {
                "image": image_path.name,
                "ground_truth": len(ground_truth),
                "predictions": len(predictions),
                "tp": image_tp,
                "fp": image_fp,
                "fn": image_fn,
                "inference_ms": elapsed * 1000,
            }
        )

        if index % 25 == 0 or index == len(images):
            print(f"Processed {index}/{len(images)}")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    avg_inference_ms = (
        sum(inference_times) / len(inference_times) * 1000 if inference_times else 0.0
    )

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    results = {
        "model": str(MODEL_PATH),
        "test_images": len(images),
        "ground_truth_objects": total_gt,
        "predictions": total_predictions,
        "confidence_threshold": CONF,
        "iou_match_threshold": IOU_MATCH,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_inference_ms": avg_inference_ms,
        "average_confidence": avg_confidence,
        "images_with_no_detection": images_with_no_detection,
        "images_with_multiple_predictions": images_with_multiple_predictions,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "v3_holdout_metrics.json").write_text(json.dumps(results, indent=2))

    (output_dir / "v3_per_image.json").write_text(json.dumps(per_image, indent=2))

    print()
    print("=" * 70)
    print("FINAL V3 HOLDOUT RESULTS")
    print("=" * 70)
    print(f"Images:                  {len(images)}")
    print(f"Ground truth objects:    {total_gt}")
    print(f"Predictions:             {total_predictions}")
    print(f"True positives:          {tp}")
    print(f"False positives:         {fp}")
    print(f"False negatives:         {fn}")
    print(f"Precision:               {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:                  {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1:                      {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"Average inference:       {avg_inference_ms:.2f} ms")
    print(f"Average confidence:      {avg_confidence:.4f}")
    print(f"No detections:           {images_with_no_detection}")
    print(f"Multiple detections:     {images_with_multiple_predictions}")
    print("=" * 70)

    print()
    print(f"Saved: {output_dir / 'v3_holdout_metrics.json'}")
    print(f"Saved: {output_dir / 'v3_per_image.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO11n V3 on the holdout set."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Directory for v3_holdout_metrics.json and v3_per_image.json.",
    )
    arguments = parser.parse_args()
    main(output_dir=arguments.output_dir)
