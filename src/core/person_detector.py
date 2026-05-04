from __future__ import annotations

import cv2
import numpy as np
import torch
from abc import ABC, abstractmethod
from scipy.spatial import distance
from tqdm import tqdm
from ultralytics import YOLO

import torchvision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

from src.core.court_reference import CourtReference
from src.core.person_detector_backend import (
    PERSON_DETECTOR_BACKEND_FASTER_RCNN_RESNET50,
    PERSON_DETECTOR_BACKEND_YOLO26X,
    normalize_person_detector_backend,
)


class PersonDetectorBase(ABC):
    """Shared court masks, player filtering, and multi-frame tracking."""

    def __init__(self, device: str):
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.court_ref = CourtReference()
        self.ref_top_court = self.court_ref.get_court_mask(2)
        self.ref_bottom_court = self.court_ref.get_court_mask(1)
        self.point_person_top = None
        self.point_person_bottom = None
        self.counter_top = 0
        self.counter_bottom = 0

    @abstractmethod
    def detect(self, image, person_min_score: float = 0.85):
        """Return (list of xyxy boxes, list of confidence scores)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def court_split_person_min_score(self) -> float:
        """Score threshold when splitting detections into top/bottom court regions."""
        raise NotImplementedError

    def detect_top_and_bottom_players(self, image, inv_matrix, filter_players=False):
        matrix = cv2.invert(inv_matrix)[1]
        mask_top_court = cv2.warpPerspective(
            self.ref_top_court, matrix, image.shape[1::-1]
        )
        mask_bottom_court = cv2.warpPerspective(
            self.ref_bottom_court, matrix, image.shape[1::-1]
        )
        person_bboxes_top, person_bboxes_bottom = [], []

        bboxes, _probs = self.detect(
            image, person_min_score=self.court_split_person_min_score
        )
        if len(bboxes) > 0:
            person_points = [
                [int((bbox[2] + bbox[0]) / 2), int(bbox[3])] for bbox in bboxes
            ]
            person_bboxes = list(zip(bboxes, person_points))

            person_bboxes_top = [
                pt
                for pt in person_bboxes
                if mask_top_court[pt[1][1] - 1, pt[1][0]] == 1
            ]
            person_bboxes_bottom = [
                pt
                for pt in person_bboxes
                if mask_bottom_court[pt[1][1] - 1, pt[1][0]] == 1
            ]

            if filter_players:
                person_bboxes_top, person_bboxes_bottom = self.filter_players(
                    person_bboxes_top, person_bboxes_bottom, matrix
                )
        return person_bboxes_top, person_bboxes_bottom

    def filter_players(self, person_bboxes_top, person_bboxes_bottom, matrix):
        """Leave one person at the top and bottom of the tennis court."""
        refer_kps = np.array(self.court_ref.key_points[12:], dtype=np.float32).reshape(
            (-1, 1, 2)
        )
        trans_kps = cv2.perspectiveTransform(refer_kps, matrix)
        center_top_court = trans_kps[0][0]
        center_bottom_court = trans_kps[1][0]
        if len(person_bboxes_top) > 1:
            dists = [
                distance.euclidean(x[1], center_top_court) for x in person_bboxes_top
            ]
            ind = dists.index(min(dists))
            person_bboxes_top = [person_bboxes_top[ind]]
        if len(person_bboxes_bottom) > 1:
            dists = [
                distance.euclidean(x[1], center_bottom_court)
                for x in person_bboxes_bottom
            ]
            ind = dists.index(min(dists))
            person_bboxes_bottom = [person_bboxes_bottom[ind]]
        return person_bboxes_top, person_bboxes_bottom

    def track_players(self, frames, matrix_all, filter_players=False):
        persons_top = []
        persons_bottom = []
        min_len = min(len(frames), len(matrix_all))
        for num_frame in tqdm(range(min_len)):
            img = frames[num_frame]
            if matrix_all[num_frame] is not None:
                inv_matrix = matrix_all[num_frame]
                person_top, person_bottom = self.detect_top_and_bottom_players(
                    img, inv_matrix, filter_players
                )
            else:
                person_top, person_bottom = [], []
            persons_top.append(person_top)
            persons_bottom.append(person_bottom)

            if num_frame > 0 and num_frame % 50 == 0:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return persons_top, persons_bottom


class FasterRcnnResnetPersonDetector(PersonDetectorBase):
    """Faster R-CNN ResNet50 FPN (COCO) — higher quality person boxes, slower than YOLO."""

    court_split_person_min_score = 0.85

    def __init__(self, device: str):
        super().__init__(device)
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        self.detection_model = fasterrcnn_resnet50_fpn(weights=weights)
        self.detection_model = self.detection_model.to(self.device)
        self.detection_model.eval()

    def detect(self, image, person_min_score: float = 0.85):
        PERSON_LABEL = 1
        frame_tensor = image.transpose((2, 0, 1)) / 255
        frame_tensor = (
            torch.from_numpy(frame_tensor).unsqueeze(0).float().to(self.device)
        )

        with torch.no_grad():
            preds = self.detection_model(frame_tensor)

        persons_boxes = []
        probs = []
        for box, label, score in zip(
            preds[0]["boxes"][:], preds[0]["labels"], preds[0]["scores"]
        ):
            if int(label) == PERSON_LABEL and float(score) > person_min_score:
                persons_boxes.append(box.detach().cpu().numpy())
                probs.append(float(score.detach().cpu().item()))

        del frame_tensor, preds
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return persons_boxes, probs


class YoloV26PersonDetector(PersonDetectorBase):
    """Ultralytics YOLO26x — faster; tuned with a lower score cutoff for court split."""

    court_split_person_min_score = 0.5

    def __init__(self, device: str):
        super().__init__(device)
        self.detection_model = YOLO("yolo26x.pt")
        if self.device.type == "cuda":
            self.detection_model.to("cuda")

    def detect(self, image, person_min_score: float = 0.85):
        PERSON_LABEL = 0
        with torch.inference_mode():
            preds = self.detection_model(image, verbose=False)[0]

        persons_boxes = []
        probs = []
        if preds.boxes is not None and len(preds.boxes) > 0:
            boxes = preds.boxes.xyxy.detach().cpu().numpy()
            labels = preds.boxes.cls.detach().cpu().numpy().astype(np.int32)
            scores = preds.boxes.conf.detach().cpu().numpy()
            for box, label, score in zip(boxes, labels, scores):
                if int(label) == PERSON_LABEL and float(score) > person_min_score:
                    persons_boxes.append(np.asarray(box, dtype=np.float32))
                    probs.append(float(score))

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return persons_boxes, probs


def build_person_detector(kind: str, device: str) -> PersonDetectorBase:
    k = normalize_person_detector_backend(kind)
    if k == PERSON_DETECTOR_BACKEND_YOLO26X:
        return YoloV26PersonDetector(device)
    return FasterRcnnResnetPersonDetector(device)


# Backwards compatibility: previous single-class name mapped to YOLO implementation.
PersonDetector = YoloV26PersonDetector
