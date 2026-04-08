import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from ultralytics import YOLO


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
YOLO26N_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt"


class FasterRCNNDetector:
    def __init__(self, device: torch.device, score_threshold: float, batch_size: int = 1) -> None:
        self.device = device
        self.score_threshold = score_threshold
        self.batch_size = max(1, int(batch_size))
        self.weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights).to(self.device).eval()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def detect_persons(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = (
            torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
        )

        with torch.inference_mode():
            preds = self.model([image_tensor])[0]

        boxes, scores = [], []
        for box, label, score in zip(preds["boxes"], preds["labels"], preds["scores"]):
            if int(label.item()) == 1 and float(score.item()) >= self.score_threshold:
                boxes.append(box.detach().cpu().numpy())
                scores.append(float(score.item()))
        return boxes, scores

    def detect_persons_batch(self, frames_bgr):
        image_tensors = []
        for frame_bgr in frames_bgr:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image_tensor = (
                torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
            )
            image_tensors.append(image_tensor)

        with torch.inference_mode():
            predictions = self.model(image_tensors)

        all_results = []
        for preds in predictions:
            boxes, scores = [], []
            for box, label, score in zip(preds["boxes"], preds["labels"], preds["scores"]):
                if int(label.item()) == 1 and float(score.item()) >= self.score_threshold:
                    boxes.append(box.detach().cpu().numpy())
                    scores.append(float(score.item()))
            all_results.append((boxes, scores))
        return all_results


class YOLO26Detector:
    def __init__(self, score_threshold: float, model_path: str, device) -> None:
        self.score_threshold = score_threshold
        self.model = YOLO(model_path)
        self.device = device

    def detect_persons(self, frame_bgr):
        results = self.model.predict(
            source=frame_bgr,
            conf=self.score_threshold,
            classes=[0],  # COCO person class
            verbose=False,
            device=self.device,
        )
        boxes, scores = [], []
        if not results:
            return boxes, scores

        result = results[0]
        if result.boxes is None or result.boxes.xyxy is None:
            return boxes, scores

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        for box, score in zip(xyxy, confs):
            boxes.append(box)
            scores.append(float(score))
        return boxes, scores


def draw_detections(frame, boxes, scores, color, model_name):
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        center_bottom = (int((x1 + x2) / 2), y2)
        cv2.circle(frame, center_bottom, 4, color, -1)
        cv2.putText(
            frame,
            f"{model_name}: {score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def process_video(video_path: Path, out_path: Path, detector, model_name: str, max_frames: int | None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_count = 0
    total_detections = 0
    inference_seconds = 0.0
    wall_start = time.perf_counter()

    batch_size = getattr(detector, "batch_size", 1)
    use_batch = hasattr(detector, "detect_persons_batch") and batch_size > 1

    while True:
        frames = []
        while len(frames) < (batch_size if use_batch else 1):
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_count + len(frames) >= max_frames:
                break
            frames.append(frame)
        if not frames:
            break

        infer_start = time.perf_counter()
        if use_batch:
            batch_results = detector.detect_persons_batch(frames)
        else:
            batch_results = [detector.detect_persons(frames[0])]
        inference_seconds += time.perf_counter() - infer_start

        for frame, (boxes, scores) in zip(frames, batch_results):
            total_detections += len(boxes)
            annotated = draw_detections(frame, boxes, scores, (0, 255, 0), model_name)
            writer.write(annotated)
            frame_count += 1

    wall_seconds = time.perf_counter() - wall_start
    cap.release()
    writer.release()

    return {
        "video": video_path.name,
        "frames_processed": frame_count,
        "total_person_detections": total_detections,
        "avg_persons_per_frame": (total_detections / frame_count) if frame_count else 0.0,
        "total_inference_seconds": inference_seconds,
        "avg_inference_ms_per_frame": (inference_seconds / frame_count * 1000.0) if frame_count else 0.0,
        "wall_seconds": wall_seconds,
        "effective_fps": (frame_count / wall_seconds) if wall_seconds else 0.0,
        "output_video": str(out_path),
    }


def summarize(metrics_per_video):
    total_frames = sum(item["frames_processed"] for item in metrics_per_video)
    total_det = sum(item["total_person_detections"] for item in metrics_per_video)
    total_inf = sum(item["total_inference_seconds"] for item in metrics_per_video)
    total_wall = sum(item["wall_seconds"] for item in metrics_per_video)
    return {
        "videos_processed": len(metrics_per_video),
        "frames_processed": total_frames,
        "total_person_detections": total_det,
        "avg_persons_per_frame": (total_det / total_frames) if total_frames else 0.0,
        "avg_inference_ms_per_frame": (total_inf / total_frames * 1000.0) if total_frames else 0.0,
        "effective_fps": (total_frames / total_wall) if total_wall else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Faster R-CNN COCO_V1 vs YOLO26 person detection on tennis_scenes videos."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("./tennis_scenes"))
    parser.add_argument("--output-dir", type=Path, default=Path("./output/person_detector_comparison"))
    parser.add_argument("--conf-thres", type=float, default=0.5)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["fasterrcnn_coco_v1", "yolo26m"],
        default=["fasterrcnn_coco_v1"],
        help="Models to run. Default runs only Faster R-CNN for quality.",
    )
    parser.add_argument(
        "--frcnn-batch-size",
        type=int,
        default=1,
        help="Faster R-CNN batch size. GTX 1060 is typically fastest at 1.",
    )
    parser.add_argument("--yolo-weights", type=str, default=YOLO26N_URL)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {args.input_dir.resolve()}")

    video_files = sorted(
        [p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    )
    if args.max_videos is not None:
        video_files = video_files[: args.max_videos]
    if not video_files:
        raise RuntimeError(f"No video files found in: {args.input_dir.resolve()}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_device = 0 if torch.cuda.is_available() else "cpu"

    models = {}
    if "fasterrcnn_coco_v1" in args.models:
        models["fasterrcnn_coco_v1"] = FasterRCNNDetector(
            device=device,
            score_threshold=args.conf_thres,
            batch_size=args.frcnn_batch_size,
        )
    if "yolo26m" in args.models:
        models["yolo26m"] = YOLO26Detector(
            score_threshold=args.conf_thres,
            model_path=args.yolo_weights,
            device=yolo_device,
        )

    comparison = {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "device": str(device),
        "confidence_threshold": args.conf_thres,
        "models": {},
    }

    for model_name, detector in models.items():
        print(f"\n=== Running {model_name} ===")
        model_output_dir = args.output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        per_video_metrics = []
        for video_path in video_files:
            output_path = model_output_dir / f"{video_path.stem}_{model_name}.mp4"
            print(f"Processing {video_path.name} -> {output_path.name}")
            metrics = process_video(
                video_path=video_path,
                out_path=output_path,
                detector=detector,
                model_name=model_name,
                max_frames=args.max_frames,
            )
            per_video_metrics.append(metrics)
            print(
                f"frames={metrics['frames_processed']}, "
                f"avg_infer_ms={metrics['avg_inference_ms_per_frame']:.2f}, "
                f"fps={metrics['effective_fps']:.2f}"
            )

        comparison["models"][model_name] = {
            "summary": summarize(per_video_metrics),
            "videos": per_video_metrics,
        }

    report_path = args.output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nSaved comparison report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
