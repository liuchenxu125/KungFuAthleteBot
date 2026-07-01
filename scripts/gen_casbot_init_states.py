"""Generate robot_init_states_8192.pth for Casbot Skeleton.

Usage:
    python scripts/gen_casbot_init_states.py
"""

import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unitree_rl_mjlab"))

from src.assets.robots.casbot_skeleton.casbot_constants import (
    get_casbot_robot_cfg,
    KNEES_BENT_KEYFRAME,
)


def resolve_joint_pos(joint_names: list[str], patterns: dict[str, float]) -> list[float]:
    """Resolve regex joint patterns to a flat list of joint positions."""
    result = [0.0] * len(joint_names)
    for pattern, value in patterns.items():
        regex = re.compile(pattern)
        for i, name in enumerate(joint_names):
            if regex.match(name):
                result[i] = value
    return result


def main():
    num_envs = 8192
    robot_cfg = get_casbot_robot_cfg()

    from mjlab.entity.entity import Entity

    robot = Entity(robot_cfg)
    joint_names = list(robot.joint_names)
    n_dof = len(joint_names)
    print(f"Robot DOF: {n_dof}")
    print(f"Joint names: {joint_names}")

    # Resolve default joint positions from the keyframe patterns
    default_pos = resolve_joint_pos(joint_names, robot_cfg.init_state.joint_pos)
    print(f"Default joint positions: {default_pos}")

    dof_pos = torch.tensor(default_pos, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    # Add small noise for diversity
    dof_pos += (torch.rand_like(dof_pos) - 0.5) * 0.1

    # Clamp to soft limits if needed
    base_height = robot_cfg.init_state.pos[2]  # ~0.844m

    # Root states: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    robot_root_states_wxyz = torch.zeros(num_envs, 13)
    robot_root_states_wxyz[:, 2] = base_height
    robot_root_states_wxyz[:, 3] = 1.0  # qw = 1

    # xyzw format
    robot_root_states_xyzw = robot_root_states_wxyz.clone()
    robot_root_states_xyzw[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])

    output_path = (
        Path(__file__).resolve().parent.parent
        / "unitree_rl_mjlab"
        / "src"
        / "assets"
        / "motions"
        / "casbot"
        / "robot_init_states_8192.pth"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "num_envs": num_envs,
        "dof_pos": dof_pos,
        "robot_root_states_wxyz": robot_root_states_wxyz,
        "robot_root_states_xyzw": robot_root_states_xyzw,
    }
    torch.save(data, str(output_path))
    print(f"\nSaved init states to: {output_path}")
    print(f"  dof_pos shape: {dof_pos.shape}")
    print(f"  robot_root_states_wxyz shape: {robot_root_states_wxyz.shape}")
    print(f"  robot_root_states_xyzw shape: {robot_root_states_xyzw.shape}")


if __name__ == "__main__":
    main()
