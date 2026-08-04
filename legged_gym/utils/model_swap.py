"""Hot-swap checkpoints of the *same* task into a live play process.

Scope and limits
----------------
Swapping a different checkpoint of the same task is a plain state-dict reload:
the network shapes, ``num_observations`` and the env's observation pipeline are
all unchanged, and ``get_inference_policy`` hands back a bound method of
``alg.actor_critic``, so reloading in place keeps the already-returned policy
callable valid.  No scene rebuild, no env rebuild.

Swapping *across methods* (mlp <-> him <-> rma <-> moects) is a different story:
those tasks have different observation dimensions and different env-side
observation assembly, so the env itself would have to be rebuilt -- and
``scene.build()`` is one-shot in Genesis.  That is deliberately NOT supported
here; :func:`load_checkpoint_into_runner` raises with an explanatory message
when the state dict does not fit.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Sequence

_MODEL_RE = re.compile(r"^model_(\d+)\.pt$")

# Named checkpoints first (they are the ones a user actually wants to compare),
# then numbered ones in training order.
_NAMED_ORDER = ("best_tracking.pt", "best.pt")


def list_checkpoints(run_dir) -> List[Path]:
    """Every checkpoint in a run directory, in a stable, meaningful order."""
    directory = Path(run_dir)
    if not directory.is_dir():
        return []
    names = {p.name for p in directory.iterdir() if p.is_file() and p.suffix == ".pt"}
    ordered: List[Path] = [directory / n for n in _NAMED_ORDER if n in names]
    numbered = []
    for name in names:
        match = _MODEL_RE.match(name)
        if match:
            numbered.append((int(match.group(1)), name))
    numbered.sort()
    ordered.extend(directory / name for _, name in numbered)
    # Anything else (custom names) last, alphabetically, so nothing is hidden.
    known = {p.name for p in ordered}
    ordered.extend(directory / n for n in sorted(names - known))
    return ordered


def load_checkpoint_into_runner(alg_runner, checkpoint_path, torch_module=None) -> str:
    """Reload ``checkpoint_path`` into a live runner.  Returns a short label.

    Raises ``ValueError`` when the checkpoint does not fit the live network --
    which is exactly the cross-method case that needs a rebuilt env.
    """
    if torch_module is None:  # injectable for tests
        import torch as torch_module  # type: ignore[no-redef]

    path = Path(checkpoint_path)
    device = getattr(alg_runner, "device", "cpu")
    checkpoint = torch_module.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict")
    if state is None:
        raise ValueError(f"{path.name}: no 'model_state_dict' in checkpoint")

    actor_critic = alg_runner.alg.actor_critic
    try:
        actor_critic.load_state_dict(state, strict=True)
    except (RuntimeError, KeyError) as exc:
        raise ValueError(
            f"{path.name} does not fit the live policy ({exc}). Hot-swap only "
            "supports checkpoints of the SAME task; changing method "
            "(mlp/him/rma/moects) changes num_observations and the env's "
            "observation pipeline, which needs a rebuilt env -- restart play.py."
        ) from exc
    actor_critic.eval()
    iteration = checkpoint.get("iter", checkpoint.get("iteration"))
    return f"{path.name}" + (f" (iter {iteration})" if iteration is not None else "")


class CheckpointSwapper:
    """Cycle through the checkpoints of one run directory at runtime."""

    def __init__(self, alg_runner, run_dir, current: Optional[str] = None) -> None:
        self.alg_runner = alg_runner
        self.run_dir = Path(run_dir)
        self.checkpoints: List[Path] = list_checkpoints(self.run_dir)
        self.index = 0
        if current:
            for i, path in enumerate(self.checkpoints):
                if path.name == os.path.basename(str(current)):
                    self.index = i
                    break

    @property
    def available(self) -> bool:
        return len(self.checkpoints) > 1

    @property
    def current(self) -> Optional[Path]:
        if not self.checkpoints:
            return None
        return self.checkpoints[self.index]

    def step(self, delta: int = 1) -> Optional[str]:
        """Load the next/previous checkpoint.  Returns a label, or None."""
        if not self.checkpoints:
            print("[arena] no checkpoints found next to the loaded run")
            return None
        if len(self.checkpoints) == 1:
            print(f"[arena] only one checkpoint in {self.run_dir.name}; nothing to swap")
            return None
        self.index = (self.index + int(delta)) % len(self.checkpoints)
        path = self.checkpoints[self.index]
        try:
            label = load_checkpoint_into_runner(self.alg_runner, path)
        except Exception as exc:
            print(f"[arena] checkpoint swap failed: {exc}")
            return None
        print(f"[arena] policy -> {label}  "
              f"({self.index + 1}/{len(self.checkpoints)})")
        return label
