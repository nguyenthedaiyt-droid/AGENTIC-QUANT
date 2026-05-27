# =============================================================================
# AGENTIC-QUANT — Unit Tests: Config Schemas
# =============================================================================

from __future__ import annotations

import pytest
from pathlib import Path
import tempfile
import yaml

from core.config.schemas import (
    MasterConfig,
    SystemConfig,
    SessionConfig,
    KillzoneConfig,
    NewsWeightsConfig,
    SystemThresholdsConfig,
    PortConfig,
)


class TestPortConfig:
    """Test PortConfig validation."""

    def test_default_ports(self):
        cfg = PortConfig()
        assert cfg.zmq_pull == 5556
        assert cfg.websocket == 47290
        assert cfg.tv_webhook == 8080
        assert cfg.prometheus == 9090

    def test_valid_custom_ports(self):
        cfg = PortConfig(zmq_pull=5560, websocket=47300)
        assert cfg.zmq_pull == 5560
        assert cfg.websocket == 47300

    def test_websocket_in_fallback_range_rejected(self):
        with pytest.raises(ValueError, match="fallback range"):
            PortConfig(websocket=47291)


class TestSystemConfig:
    """Test SystemConfig."""

    def test_default_values(self):
        cfg = SystemConfig()
        assert cfg.name == "AGENTIC-QUANT"
        assert cfg.symbol == "XAUUSD"
        assert cfg.environment == "development"
        assert cfg.backtest_mode is False

    def test_custom_environment(self):
        cfg = SystemConfig(environment="production")
        assert cfg.environment == "production"


class TestSessionConfig:
    """Test SessionConfig."""

    def test_session_defaults(self):
        cfg = SessionConfig(
            id="LONDON_OPEN_KZ",
            name="London Open Kill Zone",
            start_hour_utc=7,
            start_minute_utc=0,
            end_hour_utc=8,
            end_minute_utc=0,
            color="#3B82F6",
        )
        assert cfg.id == "LONDON_OPEN_KZ"
        assert cfg.start_hour_utc == 7
        assert cfg.end_hour_utc == 8
        # Default session_weight.ltf = 1.0 (killzone weights tu config yaml)
        assert cfg.session_weight.ltf == 1.0


class TestKillzoneConfig:
    """Test KillzoneConfig."""

    def test_get_session_by_id(self):
        cfg = KillzoneConfig(
            sessions=[
                SessionConfig(
                    id="ASIAN",
                    name="Asian",
                    start_hour_utc=22,
                    start_minute_utc=0,
                    end_hour_utc=7,
                    end_minute_utc=0,
                    color="#6B7280",
                )
            ]
        )
        session = cfg.get_session_by_id("ASIAN")
        assert session is not None
        assert session.id == "ASIAN"

    def test_get_nonexistent_session(self):
        cfg = KillzoneConfig(sessions=[])
        assert cfg.get_session_by_id("LONDON") is None


class TestNewsWeightsConfig:
    """Test NewsWeightsConfig."""

    def test_default_values(self):
        cfg = NewsWeightsConfig()
        assert cfg.alpha == 0.4
        assert cfg.pre_news_window_seconds == 900
        assert cfg.gamma_0 == 0.3

    def test_alpha_range(self):
        cfg = NewsWeightsConfig(alpha=1.5)
        assert cfg.alpha == 1.5


class TestSystemThresholdsConfig:
    """Test SystemThresholdsConfig."""

    def test_default_ic_thresholds(self):
        cfg = SystemThresholdsConfig()
        assert cfg.ic.good == 0.10
        assert cfg.ic.critical == 0.05

    def test_default_latency_thresholds(self):
        cfg = SystemThresholdsConfig()
        assert cfg.ipc_latency_ms.good == 25
        assert cfg.ipc_latency_ms.critical == 50


class TestMasterConfig:
    """Test MasterConfig factory method."""

    def test_from_empty_yaml_dir(self):
        """Khi khong co file yaml, su dung gia tri mac dinh."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MasterConfig.from_yaml_files(tmpdir)
            assert cfg.system.name == "AGENTIC-QUANT"
            assert cfg.system.symbol == "XAUUSD"

    def test_from_yaml_files_merges_killzones(self):
        """Khi co file yaml, gia tri trong yaml duoc su dung."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_content = {
                "system": {
                    "symbol": "EURUSD",
                    "environment": "staging",
                },
                "killzones": {
                    "sessions": [
                        {
                            "id": "TEST_SESSION",
                            "name": "Test",
                            "start_hour_utc": 8,
                            "start_minute_utc": 0,
                            "end_hour_utc": 10,
                            "end_minute_utc": 0,
                            "color": "#FF0000",
                        }
                    ]
                },
            }
            yaml_path = Path(tmpdir) / "system.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_content, f)

            cfg = MasterConfig.from_yaml_files(tmpdir)
            assert cfg.system.symbol == "EURUSD"
            assert cfg.system.environment == "staging"

            session = cfg.killzones.get_session_by_id("TEST_SESSION")
            assert session is not None
            assert session.id == "TEST_SESSION"
            assert session.color == "#FF0000"
