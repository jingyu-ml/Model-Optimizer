# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for recipe-agnostic calibrated Attention Grill integration."""

from __future__ import annotations

import copy
import pathlib
import sys

import pytest

pytest.importorskip("torch")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))

from qad.artifacts import _load_supported_static_anchor_recipe


class _FakeAttentionGrill:
    def __init__(self, recipe: dict, registered: tuple[str, ...]) -> None:
        self.recipe = recipe
        self.registered = registered

    def load_recipe(self, source) -> dict:
        assert source == "external-recipe"
        return copy.deepcopy(self.recipe)

    def available_types(self) -> list[str]:
        return sorted(self.registered)


@pytest.mark.parametrize("kernel", ["registered-static-a", "registered-static-b"])
def test_static_anchor_recipe_is_selected_by_capability_not_name(kernel):
    api = _FakeAttentionGrill(
        {"kernel": kernel, "smooth": {"mode": "static_anchor"}},
        registered=("registered-static-a", "registered-static-b"),
    )

    recipe = _load_supported_static_anchor_recipe(api, "external-recipe")

    assert recipe["kernel"] == kernel
    assert recipe["smooth"]["mode"] == "static_anchor"


def test_static_anchor_recipe_rejects_unregistered_kernel():
    api = _FakeAttentionGrill(
        {"kernel": "missing-static", "smooth": {"mode": "static_anchor"}},
        registered=("registered-static",),
    )

    with pytest.raises(ValueError, match="not registered"):
        _load_supported_static_anchor_recipe(api, "external-recipe")


def test_static_anchor_recipe_rejects_nonstatic_recipe():
    api = _FakeAttentionGrill(
        {"kernel": "registered-dynamic", "smooth": {"mode": None}},
        registered=("registered-dynamic",),
    )

    with pytest.raises(ValueError, match="static_anchor"):
        _load_supported_static_anchor_recipe(api, "external-recipe")
