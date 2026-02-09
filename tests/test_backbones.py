"""Tests for backbone models and registry."""

import torch

from stageguard.backbones import BACKBONE_REGISTRY, get_backbone, register_backbone
from stageguard.backbones.accusleep import AccuSleepBackbone
from stageguard.backbones.base import BackboneBase
from stageguard.backbones.usleep import USleepBackbone


class TestAccuSleepBackbone:
    def test_forward_shape(self):
        model = AccuSleepBackbone(num_classes=3, in_channels=1)
        x = torch.randn(2, 10, 1, 128)  # (B, T, C_in, samples)
        out = model(x)
        assert out.shape == (2, 10, 3)

    def test_different_num_classes(self):
        model = AccuSleepBackbone(num_classes=5)
        x = torch.randn(1, 5, 1, 64)
        assert model(x).shape == (1, 5, 5)


class TestUSleepBackbone:
    def test_forward_shape(self):
        model = USleepBackbone(num_classes=3, in_channels=1, base_filters=8, depth=2)
        x = torch.randn(2, 10, 1, 128)
        out = model(x)
        assert out.shape == (2, 10, 3)

    def test_different_config(self):
        model = USleepBackbone(num_classes=2, in_channels=2, base_filters=4, depth=2)
        x = torch.randn(1, 8, 2, 64)
        assert model(x).shape == (1, 8, 2)


class TestRegistry:
    def test_known_backbones(self):
        assert "accusleep" in BACKBONE_REGISTRY
        assert "usleep" in BACKBONE_REGISTRY

    def test_get_backbone(self):
        model = get_backbone("accusleep", num_classes=3)
        assert isinstance(model, AccuSleepBackbone)

    def test_get_backbone_unknown(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown backbone"):
            get_backbone("nonexistent")

    def test_register_custom(self):
        class MyBackbone(BackboneBase):
            def __init__(self, num_classes=3):
                super().__init__(num_classes)
                self.linear = torch.nn.Linear(4, num_classes)

            def forward(self, x):
                B, T, F = x.shape
                return self.linear(x.reshape(-1, F)).reshape(B, T, self.num_classes)

        register_backbone("custom", MyBackbone)
        assert "custom" in BACKBONE_REGISTRY
        model = get_backbone("custom", num_classes=3)
        assert isinstance(model, MyBackbone)
        # Clean up
        del BACKBONE_REGISTRY["custom"]
