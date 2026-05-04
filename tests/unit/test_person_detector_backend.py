import pytest

from src.core.person_detector_backend import (
    PERSON_DETECTOR_BACKEND_FASTER_RCNN_RESNET50,
    PERSON_DETECTOR_BACKEND_YOLO26X,
    normalize_person_detector_backend,
)


def test_normalize_canonical():
    assert (
        normalize_person_detector_backend("yolo26x")
        == PERSON_DETECTOR_BACKEND_YOLO26X
    )
    assert (
        normalize_person_detector_backend("fasterrcnn_resnet50")
        == PERSON_DETECTOR_BACKEND_FASTER_RCNN_RESNET50
    )


def test_normalize_aliases():
    assert normalize_person_detector_backend("YOLO") == PERSON_DETECTOR_BACKEND_YOLO26X
    assert (
        normalize_person_detector_backend("resnet")
        == PERSON_DETECTOR_BACKEND_FASTER_RCNN_RESNET50
    )


def test_normalize_invalid():
    with pytest.raises(ValueError):
        normalize_person_detector_backend("not_a_detector")
