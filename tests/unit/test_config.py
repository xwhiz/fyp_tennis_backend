"""Unit tests for application configuration (src.config)."""
import pytest


@pytest.mark.unit
class TestSettings:
    """Test Settings loading and defaults."""

    def test_settings_has_expected_attributes(self):
        from src.config import settings

        assert hasattr(settings, "database_url")
        assert hasattr(settings, "app_name")
        assert hasattr(settings, "app_env")
        assert hasattr(settings, "host")
        assert hasattr(settings, "port")
        assert hasattr(settings, "celery_broker_url")
        assert hasattr(settings, "celery_result_backend")
        assert hasattr(settings, "celery_app_name")
        assert hasattr(settings, "upload_chunk_size")
        assert hasattr(settings, "video_batch_size")
        assert hasattr(settings, "person_detector_backend")

    def test_database_url_is_string(self):
        from src.config import settings

        assert isinstance(settings.database_url, str)
        assert len(settings.database_url) > 0

    def test_upload_chunk_size_positive(self):
        from src.config import settings

        assert settings.upload_chunk_size > 0

    def test_port_is_int(self):
        from src.config import settings

        assert isinstance(settings.port, int)
        assert 1 <= settings.port <= 65535

    def test_app_name_non_empty(self):
        from src.config import settings

        assert isinstance(settings.app_name, str)
        assert len(settings.app_name) > 0
