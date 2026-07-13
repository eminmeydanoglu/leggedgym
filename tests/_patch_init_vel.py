"""Idempotently add the go2_bench_oracle_id_vel import + registration to
legged_gym/envs/__init__.py, anchored on the existing oracle_id lines.

Used to patch the UHeM checkout whose __init__.py does NOT carry the local
Phase-2 (rma/dreamwaq/sysid) registrations -- so we cannot just overwrite it.
"""
import io, os, sys

path = "legged_gym/envs/__init__.py"
src = io.open(path, encoding="utf-8").read()

if "go2_bench_oracle_id_vel" in src:
    print("already patched; nothing to do")
    sys.exit(0)

import_anchor = (
    "from legged_gym.envs.go2.go2_bench_oracle_id.go2_bench_oracle_id_config "
    "import Go2BenchOracleIDCfg, Go2BenchOracleIDCfgPPO\n"
)
import_add = (
    "from legged_gym.envs.go2.go2_bench_oracle_id_vel.go2_bench_oracle_id_vel "
    "import Go2BenchOracleIDVel\n"
    "from legged_gym.envs.go2.go2_bench_oracle_id_vel.go2_bench_oracle_id_vel_config "
    "import Go2BenchOracleIDVelCfg, Go2BenchOracleIDVelCfgPPO\n"
)
reg_anchor = (
    'task_registry.register("go2_bench_oracle_id", Go2BenchOracleID, '
    "Go2BenchOracleIDCfg(), Go2BenchOracleIDCfgPPO())\n"
)
reg_add = (
    'task_registry.register("go2_bench_oracle_id_vel", Go2BenchOracleIDVel, '
    "Go2BenchOracleIDVelCfg(), Go2BenchOracleIDVelCfgPPO())\n"
)

assert import_anchor in src, "import anchor (oracle_id config import) not found"
assert reg_anchor in src, "registration anchor (oracle_id register) not found"

src = src.replace(import_anchor, import_anchor + import_add, 1)
src = src.replace(reg_anchor, reg_anchor + reg_add, 1)

io.open(path, "w", encoding="utf-8").write(src)
print("patched", path)
