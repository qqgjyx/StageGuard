"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest

from stageguard.config import ModalityConfig

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


class TestModalityConfig:
    def test_from_yaml_mouse_eeg(self):
        cfg = ModalityConfig.from_yaml(CONFIGS_DIR / "mouse_eeg.yaml")
        assert cfg.num_classes == 3
        assert cfg.stage_names == ["Wake", "NREM", "REM"]
        assert cfg.epoch_sec == 4.0
        assert (0, 2) in cfg.rare_transitions
        assert len(cfg.d_min) == 3

    def test_from_yaml_actigraphy(self):
        cfg = ModalityConfig.from_yaml(CONFIGS_DIR / "actigraphy.yaml")
        assert cfg.num_classes == 2
        assert cfg.rare_transitions == []
        assert cfg.sqi_method == "acceleration_variance"

    def test_from_yaml_cardiorespiratory(self):
        cfg = ModalityConfig.from_yaml(CONFIGS_DIR / "cardiorespiratory.yaml")
        assert cfg.num_classes == 3
        assert cfg.sqi_method == "rr_interval_quality"

    def test_from_yaml_bioradar(self):
        cfg = ModalityConfig.from_yaml(CONFIGS_DIR / "bioradar.yaml")
        assert cfg.num_classes == 3
        assert cfg.sqi_method == "signal_amplitude"
        assert cfg.dataset_url is None

    def test_all_configs_parse(self):
        for yaml_file in CONFIGS_DIR.glob("*.yaml"):
            cfg = ModalityConfig.from_yaml(yaml_file)
            assert cfg.num_classes == len(cfg.stage_names)
            assert isinstance(cfg.rare_transitions, list)

    def test_validation_mismatch_num_classes(self):
        with pytest.raises(ValueError, match="num_classes"):
            ModalityConfig(
                stage_names=["Wake", "Sleep"],
                num_classes=3,
                rare_transitions=[],
            )

    def test_validation_mismatch_d_min(self):
        with pytest.raises(ValueError, match="d_min"):
            ModalityConfig(
                stage_names=["Wake", "Sleep"],
                num_classes=2,
                rare_transitions=[],
                d_min=[1, 2, 3],
            )

    def test_rare_transitions_converted_to_tuples(self, dummy_config):
        for t in dummy_config.rare_transitions:
            assert isinstance(t, tuple)
