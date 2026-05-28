#!/usr/bin/env python3
"""Test FK/IK pipeline: URDF loading, forward kinematics, inverse kinematics, round-trip.

Verifies:
  1. URDF loads from src/utils/piper_urdf/piper.urdf
  2. FK computes ee_link pose from joint angles
  3. IK solves joint angles for a target ee_link pose
  4. Round-trip: FK(IK(target)) ≈ target

Usage:
  PYTHONPATH=src:$PYTHONPATH python tests/test_kinematics.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = str(PROJECT_ROOT / "src" / "utils" / "piper_urdf" / "piper.urdf")
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

HOME_DEG = np.array([0.0, 50.60, -50.40, -1.21, 10.00, 0.0])


def test_urdf_loads():
    """Test 1: URDF loads and target frame exists."""
    from utils.kinematics import RobotKinematics

    kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="ee_link",
        joint_names=ARM_JOINTS,
    )
    assert kin.target_frame_name == "ee_link", f"wrong target frame: {kin.target_frame_name}"
    assert kin.joint_names == ARM_JOINTS, f"wrong joints: {kin.joint_names}"
    print("  PASS test_urdf_loads — target_frame='ee_link', joints=[joint1..joint6]")


def test_fk_home_pose():
    """Test 2: FK at home pose gives reasonable EE position."""
    from utils.kinematics import RobotKinematics

    kin = RobotKinematics(URDF_PATH, target_frame_name="ee_link", joint_names=ARM_JOINTS)
    T = kin.forward_kinematics(HOME_DEG)

    pos = T[:3, 3] * 1000  # m → mm
    assert T.shape == (4, 4), f"wrong shape: {T.shape}"
    assert np.all(np.isfinite(pos)), f"non-finite EE position: {pos}"
    # Home pose should put EE somewhere reachable (~300-600mm range)
    assert 100 < np.linalg.norm(pos) < 1000, f"EE position suspicious: {pos} mm"

    print(f"  PASS test_fk_home_pose — EE pos=[{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}] mm")


def test_fk_ik_roundtrip():
    """Test 3: IK from target → FK back → matches target pose."""
    from scipy.spatial.transform import Rotation
    from utils.kinematics import RobotKinematics

    kin = RobotKinematics(URDF_PATH, target_frame_name="ee_link", joint_names=ARM_JOINTS)

    # Start from home pose FK to get a valid target
    T_home = kin.forward_kinematics(HOME_DEG)
    # Perturb slightly to make a new target
    T_target = T_home.copy()
    T_target[:3, 3] += [0.02, -0.02, 0.03]  # +20, -20, +30 mm

    # IK to solve for joints
    joints_ik = kin.inverse_kinematics(HOME_DEG, T_target)
    assert len(joints_ik) == 6, f"wrong joint count: {len(joints_ik)}"

    # FK to get resulting pose
    T_result = kin.forward_kinematics(joints_ik)

    pos_err = np.linalg.norm(T_target[:3, 3] - T_result[:3, 3])
    rot_err = np.linalg.norm(
        Rotation.from_matrix(T_target[:3, :3]).as_rotvec()
        - Rotation.from_matrix(T_result[:3, :3]).as_rotvec()
    )

    print(f"  PASS test_fk_ik_roundtrip — pos_err={pos_err*1e3:.2f}mm, rot_err={np.rad2deg(rot_err):.3f}deg")
    assert pos_err < 1e-3, f"position error too large: {pos_err*1e3:.2f}mm"
    assert rot_err < 0.02, f"rotation error too large: {np.rad2deg(rot_err):.3f}deg"


def test_ik_multiple_targets():
    """Test 4: IK solves for multiple nearby targets without errors."""
    from scipy.spatial.transform import Rotation
    from utils.kinematics import RobotKinematics

    kin = RobotKinematics(URDF_PATH, target_frame_name="ee_link", joint_names=ARM_JOINTS)
    T_home = kin.forward_kinematics(HOME_DEG)
    current_joints = HOME_DEG.copy()

    offsets = [
        [0.01, 0, 0],
        [-0.01, 0, 0],
        [0, 0.01, 0],
        [0, -0.01, 0],
        [0, 0, 0.01],
        [0, 0, -0.01],
    ]

    for i, offset in enumerate(offsets):
        T_target = T_home.copy()
        T_target[:3, 3] += offset

        joints_ik = kin.inverse_kinematics(current_joints, T_target)
        T_result = kin.forward_kinematics(joints_ik)

        pos_err = np.linalg.norm(T_target[:3, 3] - T_result[:3, 3])
        assert pos_err < 1e-3, f"offset {offset}: pos_err={pos_err*1e3:.2f}mm (too large)"
        current_joints = joints_ik

    print(f"  PASS test_ik_multiple_targets — {len(offsets)} targets all converged")


def main():
    print("Testing FK/IK pipeline:")
    print(f"  URDF: {URDF_PATH}")
    print(f"  Target frame: ee_link")
    print()

    try:
        test_urdf_loads()
        test_fk_home_pose()
        test_fk_ik_roundtrip()
        test_ik_multiple_targets()
    except Exception as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)

    print("\nAll FK/IK tests passed.")


if __name__ == "__main__":
    main()
