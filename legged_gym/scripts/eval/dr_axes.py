"""OOD axis definitions -- a uniform interface for sweeping any physics parameter.

Each axis knows how to (a) push a per-env vector of values into the simulator and
(b) advertise its in-distribution range (for annotating where OOD begins) plus a
sensible default sweep grid. Terrain will plug in here later behind the same
interface (plan: "zemin de bir OOD ekseni olacak sekilde tasarla").

The setters reach into Genesis simulator internals (`env.simulator._robot`, the
`_friction_values` / `_added_base_mass` / `_base_com_bias` buffers). These are the
same private handles `_randomize_friction` / `_randomize_base_mass` /
`_randomize_com_displacement` use; there is no public per-env setter, and the eval
harness is Genesis-only for now (Faz 1).

Important: friction / base-mass / com are applied at BUILD time only and are NOT
re-drawn on episode reset, so writing them once after build is stable for the
whole run -- exactly the per-env fixed "true parameter" we want.

EVAL ISOLATION INVARIANT
------------------------
The oracle's privileged observation P = [friction(1), added_mass(1), com_bias(3)]
is read directly from simulator buffers.  Some of those buffers carry nonzero
stale values after build (e.g. `_added_base_mass` initialises to 1.0, NOT 0.0).
To guarantee a clean sweep, EVERY privileged dimension that the oracle reads MUST
be registered in PRIVILEGED_PARAMS below with a setter and a nominal value.
pin_others_to_nominal() pins all non-swept dims to their nominals, touching both
the physics and the label buffer.  If you add a new privileged dim to the oracle's
P, you MUST add it here -- otherwise eval will silently read stale labels.
"""

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np
import torch


def _set_friction(env, values: torch.Tensor):
    """Set per-env ground/robot contact friction ratio. values: (num_envs,)."""
    sim = env.simulator
    robot = sim._robot
    n_links = robot.n_links
    env_ids = torch.arange(env.num_envs, device=env.device)
    ratios = values.view(-1, 1).repeat(1, n_links)  # (num_envs, n_links)
    robot.set_friction_ratio(ratios, torch.arange(0, n_links), env_ids)
    sim._friction_values[env_ids] = values.view(-1, 1).to(sim._friction_values.dtype).clone()


def _set_added_mass(env, values: torch.Tensor):
    """Set per-env added base mass [kg]. values: (num_envs,)."""
    sim = env.simulator
    env_ids = torch.arange(env.num_envs, device=env.device)
    added = values.view(-1, 1).to(sim._added_base_mass.dtype)
    sim._robot.set_mass_shift(added, sim._base_link_index, env_ids)
    sim._added_base_mass[env_ids] = added.clone()


def _set_com_bias(env, values):
    """Pin per-env base CoM displacement [m].

    values may be:
      - a scalar  -> broadcast to (num_envs, 3), all three components equal
      - a 1-D (num_envs,) tensor -> each env gets [v, v, v]
      - a 2-D (num_envs, 3) tensor -> used directly

    Mirrors _randomize_com_displacement in genesis_simulator.py:
      build disp of shape (num_envs, 1, 3), write _base_com_bias, then call
      set_COM_shift.
    """
    sim = env.simulator
    env_ids = torch.arange(env.num_envs, device=env.device)
    n = env.num_envs

    if isinstance(values, (int, float)):
        xyz = torch.full((n, 3), float(values), device=env.device, dtype=torch.float)
    else:
        t = values if isinstance(values, torch.Tensor) else torch.tensor(values, device=env.device, dtype=torch.float)
        if t.dim() == 0 or t.numel() == 1:
            xyz = t.expand(n).unsqueeze(1).expand(n, 3).contiguous().float()
        elif t.dim() == 1 and t.numel() == n:
            xyz = t.unsqueeze(1).expand(n, 3).contiguous().float()
        elif t.dim() == 2 and t.shape == (n, 3):
            xyz = t.float()
        else:
            raise ValueError(
                f"_set_com_bias: expected scalar, (num_envs,), or (num_envs, 3) tensor; got {tuple(t.shape)}"
            )

    disp = xyz.unsqueeze(1)                          # (num_envs, 1, 3)
    sim._base_com_bias[env_ids] = xyz.clone()
    sim._robot.set_COM_shift(disp, sim._base_link_index, env_ids)


def _set_com_component(component: int):
    """Build an isolated CoM-axis setter; non-swept components stay nominal."""
    def setter(env, values: torch.Tensor):
        xyz = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float)
        xyz[:, component] = values
        _set_com_bias(env, xyz)
    return setter


@dataclass
class Axis:
    name: str
    setter: Callable[[object, torch.Tensor], None]
    in_dist: tuple            # (lo, hi) training range -> shade OOD outside this
    nominal: float            # value at which this axis is "off"/neutral
    default_grid: List[float] = field(default_factory=list)
    unit: str = ""

    def grid(self) -> np.ndarray:
        return np.asarray(self.default_grid, dtype=np.float64)


# --- registry -------------------------------------------------------------
# in_dist ranges mirror Go2BenchmarkCommonCfg.domain_rand.
AXES = {
    "friction": Axis(
        name="friction",
        setter=_set_friction,
        in_dist=(0.5, 1.25),
        nominal=1.0,          # unit friction ratio
        # spans well below and well above the training band to expose OOD collapse
        default_grid=[0.1, 0.25, 0.4, 0.5, 0.7, 0.9, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5],
        unit="ratio",
    ),
    "added_mass": Axis(
        name="added_mass",
        setter=_set_added_mass,
        in_dist=(-1.0, 1.0),
        nominal=0.0,          # no added mass
        default_grid=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        unit="kg",
    ),
    "com_x": Axis(
        name="com_x", setter=_set_com_component(0), in_dist=(-0.03, 0.03),
        nominal=0.0, default_grid=[-0.08, -0.05, -0.03, 0.0, 0.03, 0.05, 0.08], unit="m",
    ),
    "com_y": Axis(
        name="com_y", setter=_set_com_component(1), in_dist=(-0.03, 0.03),
        nominal=0.0, default_grid=[-0.08, -0.05, -0.03, 0.0, 0.03, 0.05, 0.08], unit="m",
    ),
    "com_z": Axis(
        name="com_z", setter=_set_com_component(2), in_dist=(-0.03, 0.03),
        nominal=0.0, default_grid=[-0.08, -0.05, -0.03, 0.0, 0.03, 0.05, 0.08], unit="m",
    ),
}


def get_axis(name: str) -> Axis:
    if name not in AXES:
        raise KeyError(f"Unknown OOD axis '{name}'. Available: {list(AXES)}")
    return AXES[name]


# --- privileged-param registry -------------------------------------------
# EVERY dimension of the oracle's privileged observation P must appear here.
# P = [friction(1), added_mass(1), com_bias(3)].
#
# Each entry maps a privileged-param name to (setter_callable, nominal_value).
# com_bias is a 3-vector; its setter accepts a scalar and broadcasts to all
# three components.  friction and added_mass are scalars.
#
# AXIS_TO_PRIV maps a sweep-axis name to its privileged-param name so
# pin_others_to_nominal can skip the correct entry.
#
# INVARIANT: if you add a new dim to the oracle's P, add it here AND in
# AXIS_TO_PRIV (if it is also a sweep axis) -- otherwise a sweep will
# silently read a stale or uninitialised label for that dimension.
_PRIVILEGED_PARAMS = {
    # priv_name: (setter, nominal)
    "friction":   (_set_friction,   1.0),   # buffer inits to 0; friction nominal=1
    "added_mass": (_set_added_mass, 0.0),   # buffer inits to 1; mass nominal=0
    "com_bias":   (_set_com_bias,   0.0),   # buffer inits to 0; COM nominal=0
}

# Maps each sweep-axis name to its corresponding privileged-param name.
_AXIS_TO_PRIV = {
    "friction":   "friction",
    "added_mass": "added_mass",
    "com_x":      "com_bias",
    "com_y":      "com_bias",
    "com_z":      "com_bias",
}


def pin_others_to_nominal(env, swept: str):
    """Force every non-swept privileged dimension to its nominal value.

    Isolation needs more than disabling domain-rand flags: some simulator buffers
    carry nonzero stale values after build (e.g. `_added_base_mass` inits to 1.0,
    NOT 0.0).  Explicitly setting every non-swept privileged dim pins BOTH the
    physics and the label buffer, so the oracle's P is never corrupted.

    INVARIANT: every privileged dimension that the oracle reads must be registered
    in _PRIVILEGED_PARAMS.  If a future dimension is not registered here, eval will
    silently read whatever value the simulator left in that buffer.

    Raises KeyError if `swept` is not a recognised sweep axis (fail-loud contract).
    """
    if swept not in _AXIS_TO_PRIV:
        raise KeyError(
            f"pin_others_to_nominal: '{swept}' is not a known sweep axis. "
            f"Known axes: {list(_AXIS_TO_PRIV)}. "
            "If you added a new axis, also register it in _AXIS_TO_PRIV (and "
            "_PRIVILEGED_PARAMS if it has a privileged dimension)."
        )
    swept_priv = _AXIS_TO_PRIV[swept]

    for priv_name, (setter, nominal) in _PRIVILEGED_PARAMS.items():
        if priv_name == swept_priv:
            continue
        if priv_name == "com_bias":
            # _set_com_bias accepts a scalar and broadcasts to (num_envs, 3)
            setter(env, nominal)
        else:
            # scalar-axis setters expect a (num_envs,) float tensor
            vals = torch.full((env.num_envs,), float(nominal), device=env.device)
            setter(env, vals)
