"""Unit tests for Pydantic schemas (src.schemas)."""
from datetime import datetime
import pytest


@pytest.mark.unit
class TestProcessVideoResponse:
    """Test ProcessVideoResponse schema."""

    def test_valid_response(self):
        from src.schemas.process_video_response import ProcessVideoResponse

        data = ProcessVideoResponse(
            success=True,
            message="OK",
            data={"process_id": "1", "filename": "x.mp4"},
        )
        assert data.success is True
        assert data.message == "OK"
        assert data.data["process_id"] == "1"

    def test_missing_success_raises(self):
        from src.schemas.process_video_response import ProcessVideoResponse
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProcessVideoResponse(message="OK", data={})


@pytest.mark.unit
class TestVideoPathsSchema:
    """Test VideoPathsSchema."""

    def test_valid_schema(self):
        from src.schemas.video_paths import VideoPathsSchema

        now = datetime.now()
        s = VideoPathsSchema(
            id=1,
            task_id=1,
            name="test",
            output_path="/out.mp4",
            minimap_path="/mini.png",
            created_at=now,
            updated_at=now,
        )
        assert s.task_id == 1
        assert s.output_path == "/out.mp4"


@pytest.mark.unit
class TestBallTrackSchema:
    """Test BallTrackSchema."""

    def test_valid_schema(self):
        from src.schemas.ball_track import BallTrackSchema

        now = datetime.now()
        s = BallTrackSchema(
            id=1,
            task_id=1,
            ball_track={"0": [100.0, 200.0], "1": [101.0, 201.0]},
            created_at=now,
            updated_at=now,
        )
        assert s.ball_track["0"] == [100.0, 200.0]


@pytest.mark.unit
class TestBouncesSchema:
    """Test BouncesSchema."""

    def test_valid_schema(self):
        from src.schemas.bounces import BouncesSchema

        now = datetime.now()
        s = BouncesSchema(
            id=1,
            task_id=1,
            bounces={"10": {"position": [100.0, 200.0], "serve": True}},
            created_at=now,
            updated_at=now,
        )
        assert s.bounces["10"]["serve"] is True


@pytest.mark.unit
class TestThumbnailSchema:
    """Test ThumbnailSchema."""

    def test_valid_schema(self):
        from src.schemas.thumbnail import ThumbnailSchema

        now = datetime.now()
        s = ThumbnailSchema(
            id=1,
            task_id=1,
            thumbnail_path="/thumb.jpg",
            created_at=now,
            updated_at=now,
        )
        assert s.thumbnail_path == "/thumb.jpg"


@pytest.mark.unit
class TestDirectionChangeIndicesSchema:
    """Test DirectionChangeIndicesSchema."""

    def test_valid_schema(self):
        from src.schemas.direction_change_indices import DirectionChangeIndicesSchema

        now = datetime.now()
        s = DirectionChangeIndicesSchema(
            id=1,
            task_id=1,
            direction_change_indices={"0": [1.0, 2.0]},
            created_at=now,
            updated_at=now,
        )
        assert s.direction_change_indices["0"] == [1.0, 2.0]


@pytest.mark.unit
class TestPlayerPositionsSchema:
    """Test PlayerPositionsSchema."""

    def test_valid_schema(self):
        from src.schemas.player_positions import PlayerPositionsSchema

        now = datetime.now()
        s = PlayerPositionsSchema(
            id=1,
            task_id=1,
            positions={"0": {"top": [1, 2, 3, 4], "bottom": None}},
            created_at=now,
            updated_at=now,
        )
        assert s.positions["0"]["top"] == [1, 2, 3, 4]
