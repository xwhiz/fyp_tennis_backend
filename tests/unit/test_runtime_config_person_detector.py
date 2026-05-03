from unittest.mock import MagicMock, patch

import pytest

from src.core.person_detector_backend import (
    PERSON_DETECTOR_BACKEND_FASTER_RCNN_RESNET50,
    PERSON_DETECTOR_BACKEND_YOLO26X,
)
from src.services import runtime_config as rc


def test_get_active_person_detector_backend_no_redis(monkeypatch):
    monkeypatch.setattr(rc.settings, "person_detector_backend", "fasterrcnn_resnet50")
    with patch.object(rc, "_redis_client", return_value=None):
        assert (
            rc.get_active_person_detector_backend()
            == PERSON_DETECTOR_BACKEND_FASTER_RCNN_RESNET50
        )


def test_set_active_person_detector_backend_no_redis():
    with patch.object(rc, "_redis_client", return_value=None):
        with pytest.raises(RuntimeError):
            rc.set_active_person_detector_backend("yolo26x")


def test_get_active_reads_redis(monkeypatch):
    monkeypatch.setattr(rc.settings, "person_detector_backend", "fasterrcnn_resnet50")
    fake = MagicMock()
    fake.get.return_value = PERSON_DETECTOR_BACKEND_YOLO26X
    with patch.object(rc, "_redis_client", return_value=fake):
        assert rc.get_active_person_detector_backend() == PERSON_DETECTOR_BACKEND_YOLO26X


def test_get_active_seeds_redis_when_key_missing(monkeypatch):
    monkeypatch.setattr(rc.settings, "person_detector_backend", "yolo26x")
    fake = MagicMock()
    fake.get.return_value = None
    with patch.object(rc, "_redis_client", return_value=fake):
        out = rc.get_active_person_detector_backend()
    assert out == PERSON_DETECTOR_BACKEND_YOLO26X
    fake.set.assert_called_once()


def test_set_active_writes_redis():
    fake = MagicMock()
    with patch.object(rc, "_redis_client", return_value=fake):
        out = rc.set_active_person_detector_backend("fasterrcnn_resnet50")
    assert out == PERSON_DETECTOR_BACKEND_FASTER_RCNN_RESNET50
    fake.set.assert_called_once()
