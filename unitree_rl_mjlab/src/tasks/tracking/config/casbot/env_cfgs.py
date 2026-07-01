"""Casbot Skeleton (25 DOF) flat tracking environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from src.assets.robots.casbot_skeleton.casbot_constants import (
    CASBOT_ACTION_SCALE,
    get_casbot_robot_cfg,
)
from src.tasks.tracking.mdp import MotionStandingCommandCfg
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from src.tasks.tracking.tracking_standing_env_cfg import *


def casbot_flat_tracking_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Casbot Skeleton flat terrain tracking configuration."""
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": get_casbot_robot_cfg()}

    # Self-collision sensor: use casbot's root body name "base_link"
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    # Action scale from casbot motor parameters
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = CASBOT_ACTION_SCALE

    # Motion command: map G1 body names → casbot body names
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    # Casbot has waist_yaw_link as its upper body (no separate torso_link)
    motion_cmd.anchor_body_name = "waist_yaw_link"
    motion_cmd.body_names = (
        "base_link",
        "left_leg_pelvic_roll_link",
        "left_leg_knee_pitch_link",
        "left_leg_ankle_roll_link",
        "right_leg_pelvic_roll_link",
        "right_leg_knee_pitch_link",
        "right_leg_ankle_roll_link",
        "waist_yaw_link",
        "left_shoulder_roll_link",
        "left_elbow_pitch_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_pitch_link",
        "right_wrist_yaw_link",
    )

    # Foot collision geoms: casbot has single box collision per foot
    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r"^(left|right)_foot_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("waist_yaw_link",)

    # End-effector body names for termination checks
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_leg_ankle_roll_link",
        "right_leg_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )

    cfg.viewer.body_name = "waist_yaw_link"

    # Modify observations if we don't have state estimation
    if not has_state_estimation:
        new_actor_terms = {
            k: v
            for k, v in cfg.observations["actor"].terms.items()
            if k not in ["motion_anchor_pos_b", "base_lin_vel"]
        }
        cfg.observations["actor"] = ObservationGroupCfg(
            terms=new_actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        )

    # Apply play mode overrides
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.sampling_mode = "start"

    return cfg


def _post_process_casbot_flat_tracking_standing_env_cfg(
    cfg: ManagerBasedRlEnvCfg,
    has_state_estimation: bool,
    play: bool,
) -> ManagerBasedRlEnvCfg:

    cfg.scene.entities = {"robot": get_casbot_robot_cfg()}

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = CASBOT_ACTION_SCALE

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionStandingCommandCfg)
    motion_cmd.init_pos_file = "src/assets/motions/casbot/robot_init_states_8192.pth"
    motion_cmd.anchor_body_name = "waist_yaw_link"
    motion_cmd.root_body_name = ("base_link",)
    motion_cmd.shoulders_body_names = (
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
    )
    motion_cmd.feet_body_names = (
        "left_leg_ankle_roll_link",
        "right_leg_ankle_roll_link",
    )
    motion_cmd.body_names = (
        "base_link",
        "left_leg_pelvic_roll_link",
        "left_leg_knee_pitch_link",
        "left_leg_ankle_roll_link",
        "right_leg_pelvic_roll_link",
        "right_leg_knee_pitch_link",
        "right_leg_ankle_roll_link",
        "waist_yaw_link",
        "left_shoulder_roll_link",
        "left_elbow_pitch_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_pitch_link",
        "right_wrist_yaw_link",
    )

    # Foot collision geoms
    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r"^(left|right)_foot_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("waist_yaw_link",)

    cfg.viewer.body_name = "waist_yaw_link"

    # Modify observations if no state estimation
    if not has_state_estimation:
        new_actor_terms = {
            k: v
            for k, v in cfg.observations["actor"].terms.items()
            if k not in ["motion_anchor_pos_b", "base_lin_vel"]
        }
        cfg.observations["actor"] = ObservationGroupCfg(
            terms=new_actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        )

    # Apply play mode overrides
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.terminations.pop("tracking_failure", None)
        cfg.commands["motion"].tracking_standing_weight = (0.0, 1.0)

        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.sampling_mode = "start"

    return cfg


def casbot_flat_tracking_standing_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_tracking_standing_env_cfg()
    # Update termination body names for casbot
    for name, _func, params in cfg.terminations["tracking_failure"].func.terms:
        if name == "ee_body_pos_z":
            params["body_names"] = (
                "left_leg_ankle_roll_link",
                "right_leg_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            )
            break
    # Update knee joint pattern for electrical power cost
    cfg.rewards["electrical_power_cost"].params[
        "asset_cfg"
    ].joint_names = (".*_leg_knee_pitch_joint",)
    cfg = _post_process_casbot_flat_tracking_standing_env_cfg(
        cfg, has_state_estimation, play
    )
    return cfg


def casbot_flat_tracking_standing_env_cfg_1307_stage_I(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_tracking_standing_env_cfg_1307_stage_I()
    for name, _func, params in cfg.terminations["tracking_failure"].func.terms:
        if name == "ee_body_pos_z":
            params["body_names"] = (
                "left_leg_ankle_roll_link",
                "right_leg_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            )
            break
    cfg.rewards["electrical_power_cost"].params[
        "asset_cfg"
    ].joint_names = (".*_leg_knee_pitch_joint",)
    cfg = _post_process_casbot_flat_tracking_standing_env_cfg(
        cfg, has_state_estimation, play
    )
    return cfg


def casbot_flat_tracking_standing_env_cfg_1307_stage_II(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_tracking_standing_env_cfg_1307_stage_II()
    cfg.rewards["electrical_power_cost"].params[
        "asset_cfg"
    ].joint_names = (".*_leg_knee_pitch_joint",)
    cfg = _post_process_casbot_flat_tracking_standing_env_cfg(
        cfg, has_state_estimation, play
    )
    return cfg


def casbot_flat_tracking_standing_env_cfg_1307_stage_III(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_tracking_standing_env_cfg_1307_stage_III()
    cfg.rewards["electrical_power_cost"].params[
        "asset_cfg"
    ].joint_names = (".*_leg_knee_pitch_joint",)
    cfg = _post_process_casbot_flat_tracking_standing_env_cfg(
        cfg, has_state_estimation, play
    )
    return cfg
