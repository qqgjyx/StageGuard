"""Backbone registry and factory."""

from __future__ import annotations

from typing import Dict, Type

from .accusleep import AccuSleepBackbone
from .base import BackboneBase
from .usleep import USleepBackbone

BACKBONE_REGISTRY: Dict[str, Type[BackboneBase]] = {
    "accusleep": AccuSleepBackbone,
    "usleep": USleepBackbone,
}


def get_backbone(name: str, **kwargs) -> BackboneBase:
    """Instantiate a backbone by name.

    Args:
        name: Backbone name (must be in BACKBONE_REGISTRY).
        **kwargs: Arguments forwarded to the backbone constructor.

    Returns:
        Instantiated backbone module.
    """
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Available: {list(BACKBONE_REGISTRY.keys())}"
        )
    return BACKBONE_REGISTRY[name](**kwargs)


def register_backbone(name: str, cls: Type[BackboneBase]) -> None:
    """Register a custom backbone class."""
    BACKBONE_REGISTRY[name] = cls
