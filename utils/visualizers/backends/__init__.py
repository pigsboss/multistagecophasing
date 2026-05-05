# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any
import sys

# Forward reference
if "..base" not in sys.modules:
    from ..base import Scene  # pragma: no cover


class Renderer(ABC):
    """Abstract base for all visualization backends."""

    @abstractmethod
    def render(self, scene: Scene, **kwargs: Any) -> None:
        """Render the given scene."""
        ...
