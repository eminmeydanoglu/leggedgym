from .simulator import Simulator
from .genesis_simulator import GenesisSimulator
from .isaacgym_simulator import IsaacGymSimulator
from .isaaclab_simulator import IsaacLabSimulator

# MuJoCo is an optional sim-to-sim backend: the module may be missing entirely
# (it is landing separately) and even when present it needs the `mujoco` wheel,
# which is not part of the base install. Importing this package must never break
# for genesis/isaacgym users just because of that, so failures degrade to None
# and BaseTask raises a pointed error only if someone actually selects mujoco.
try:
    from .mujoco_simulator import MujocoSimulator
except ImportError:
    MujocoSimulator = None