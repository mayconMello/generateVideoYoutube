"""Shared runtime configuration for src package."""

from __future__ import annotations

import os

# Keep torchvision-heavy utilities disabled unless explicitly needed.
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("DISABLE_TRANSFORMERS_AV_IMPORTS", "1")
