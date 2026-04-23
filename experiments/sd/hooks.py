"""Activation-hooking utilities for capturing intermediate representations."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn


class ActivationRecorder:
    """Register forward hooks on named modules and store their outputs.

    Usage:
        recorder = ActivationRecorder(model, layer_names=["layer3", "layer5"])
        with recorder:
            out = model(x)
        acts = recorder.activations  # OrderedDict[str, Tensor]
    """

    def __init__(self, model: nn.Module, layer_names: list[str] | None = None):
        self.model = model
        self.layer_names = layer_names
        self.activations: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    # -- context-manager interface ------------------------------------------
    def __enter__(self) -> "ActivationRecorder":
        self.activations.clear()
        targets = self._resolve_layers()
        for name, module in targets:
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)
        return self

    def __exit__(self, *_: object) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # -- internals -----------------------------------------------------------
    def _resolve_layers(self) -> list[tuple[str, nn.Module]]:
        named = dict(self.model.named_modules())
        if self.layer_names is None:
            return list(named.items())
        out: list[tuple[str, nn.Module]] = []
        for name in self.layer_names:
            if name not in named:
                raise KeyError(
                    f"Module '{name}' not found. Available: "
                    f"{list(named.keys())[:20]}…"
                )
            out.append((name, named[name]))
        return out

    def _make_hook(self, name: str) -> Callable:
        def hook(_module: nn.Module, _input: object, output: torch.Tensor) -> None:
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach()
            elif isinstance(output, (tuple, list)):
                self.activations[name] = output[0].detach()
        return hook
