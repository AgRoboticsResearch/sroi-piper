#!/usr/bin/env python3
"""Benchmark IK computation FPS — isolates solver speed, no hardware.

Usage:
  PYTHONPATH=src:$PYTHONPATH python tests/test_ik_bench.py
  PYTHONPATH=src:$PYTHONPATH python tests/test_ik_bench.py --iterations 10000 --max_ik_iters 20
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = str(PROJECT_ROOT / "src" / "utils" / "piper_urdf" / "piper.urdf")
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

HOME_DEG = np.array([0.0, 50.60, -50.40, -1.21, 10.00, 0.0])
RNG = np.random.default_rng(42)


def random_target_near(T_ref, pos_std=0.01, rot_std=0.05):
    T = T_ref.copy()
    T[:3, 3] += RNG.normal(0, pos_std, 3)
    rand_rotvec = RNG.normal(0, rot_std, 3)
    T[:3, :3] = T[:3, :3] @ Rotation.from_rotvec(rand_rotvec).as_matrix()
    return T


class IKBench:
    """Solver with configurable max iterations for benchmarking."""

    def __init__(self, urdf_path, target_frame_name, joint_names):
        import placo

        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.target_frame_name = target_frame_name
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names
        self._tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

    def fk(self, joint_deg):
        joint_rad = np.deg2rad(joint_deg[:len(self.joint_names)])
        for i, jn in enumerate(self.joint_names):
            self.robot.set_joint(jn, joint_rad[i])
        self.robot.update_kinematics()
        return self.robot.get_T_world_frame(self.target_frame_name)

    def ik(self, current_joints_deg, T_target, pos_w=1.0, rot_w=0.01, max_iters=50):
        current_rad = np.deg2rad(current_joints_deg[:len(self.joint_names)])
        for i, jn in enumerate(self.joint_names):
            self.robot.set_joint(jn, current_rad[i])

        self._tip_frame.T_world_frame = T_target
        self._tip_frame.configure(self.target_frame_name, "soft", pos_w, rot_w)

        for i in range(max_iters):
            self.solver.solve(True)
            self.robot.update_kinematics()
            current_T = self.robot.get_T_world_frame(self.target_frame_name)
            err = np.linalg.norm(T_target[:3, 3] - current_T[:3, 3])
            if err < 1e-4:
                break

        joint_rad = np.array([self.robot.get_joint(jn) for jn in self.joint_names])
        joint_deg = np.rad2deg(joint_rad)
        return joint_deg, i + 1


def main():
    parser = argparse.ArgumentParser(description="Benchmark IK computation FPS")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--max_ik_iters", type=int, default=50,
                        help="Max IK solver iterations per call")
    parser.add_argument("--pos_std", type=float, default=0.01,
                        help="Position perturbation std dev (m)")
    parser.add_argument("--rot_std", type=float, default=0.05,
                        help="Rotation perturbation std dev (rad)")
    parser.add_argument("--position_weight", type=float, default=1.0)
    parser.add_argument("--orientation_weight", type=float, default=0.01)
    args = parser.parse_args()

    print(f"Loading URDF: {URDF_PATH}")
    kin = IKBench(URDF_PATH, target_frame_name="ee_link", joint_names=ARM_JOINTS)

    T_home = kin.fk(HOME_DEG)

    # Warm-up
    for _ in range(100):
        T = random_target_near(T_home, args.pos_std, args.rot_std)
        kin.ik(HOME_DEG, T, args.position_weight, args.orientation_weight, args.max_ik_iters)

    # Generate targets
    targets = [random_target_near(T_home, args.pos_std, args.rot_std) for _ in range(args.iterations)]

    # Benchmark
    joints = HOME_DEG.copy()
    errors = []
    iters_used = []
    failures = 0

    t_start = time.perf_counter()
    for i in range(args.iterations):
        joints, n_iters = kin.ik(joints, targets[i],
                                 args.position_weight, args.orientation_weight,
                                 args.max_ik_iters)
        T_result = kin.fk(joints)
        pos_err = np.linalg.norm(targets[i][:3, 3] - T_result[:3, 3])
        errors.append(pos_err * 1000)  # mm
        iters_used.append(n_iters)
        if pos_err > 1e-3:
            failures += 1
    elapsed = time.perf_counter() - t_start

    fps = args.iterations / elapsed
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    p99_err = np.percentile(errors, 99)
    mean_iters = np.mean(iters_used)
    p99_iters = np.percentile(iters_used, 99)
    fail_pct = failures / args.iterations * 100

    print()
    print(f"Iterations:  {args.iterations}")
    print(f"Max IK iter: {args.max_ik_iters}")
    print(f"Total time:  {elapsed:.3f} s")
    print(f"IK FPS:      {fps:.0f} Hz  (avg latency: {elapsed/args.iterations*1e6:.1f} us)")
    print(f"IK iters:    mean={mean_iters:.1f}  p99={p99_iters:.0f}")
    print(f"Pos error:   mean={mean_err:.3f} mm  max={max_err:.2f} mm  p99={p99_err:.3f} mm")
    print(f"Fail (>1mm): {failures}/{args.iterations} ({fail_pct:.1f}%)")
    print(f"Weights:     pos={args.position_weight}  orient={args.orientation_weight}")
    print(f"Perturb:     pos_std={args.pos_std*1000:.0f}mm  rot_std={np.rad2deg(args.rot_std):.1f}deg")


if __name__ == "__main__":
    main()
