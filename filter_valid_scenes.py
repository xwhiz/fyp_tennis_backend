from pathlib import Path
import shutil
import cv2
from src.core.court_detection_net import CourtDetectorNet
import torch

from src.core.process_video import get_valid_scenes
from src.core.utils import scene_detect
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
court_detector = CourtDetectorNet("./src/model_tennis_court_det.pt", DEVICE)


def unique_destination_path(destination_dir: Path, source_name: str) -> Path:
    destination = destination_dir / source_name
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    counter = 1
    while True:
        candidate = destination_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def main() -> None:
    source_dir = Path("./clean_videos")
    destination_dir = Path("./tennis_scenes")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_dir.resolve()}")

    destination_dir.mkdir(parents=True, exist_ok=True)

    for video_file in sorted(source_dir.iterdir()):
        if not video_file.is_file():
            continue

        video_path = str(video_file)
        scenes = scene_detect(video_path)
        print(f"Scenes: {scenes}")
        valid_scenes = get_valid_scenes(court_detector, video_path, scenes)
        print(f"Valid scenes: {valid_scenes}")

        if not valid_scenes:
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Skipping unreadable video: {video_file.name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

        for scene_index, (start, end) in enumerate(valid_scenes, start=1):
            start_frame = int(start)
            end_frame = int(end)
            if end_frame <= start_frame:
                continue

            output_name = f"{video_file.stem}_scene_{scene_index}_{start_frame}-{end_frame}.mp4"
            output_path = unique_destination_path(destination_dir, output_name)
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)

            writer.release()
            print(f"Saved valid scene: {output_path.name}")

        cap.release()


if __name__ == "__main__":
    main()
