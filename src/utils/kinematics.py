"""Robot kinematics using placo for FK and IK (LeRobot-free).

Replaces lerobot.model.kinematics.RobotKinematics.
"""

import numpy as np


class RobotKinematics:
    """Thin wrapper around placo for forward and inverse kinematics.

    Usage:
        kin = RobotKinematics(urdf_path, target_frame_name="ee_link",
                              joint_names=["joint1", ..., "joint6"])
        T = kin.forward_kinematics(joints_deg)
        targets = kin.inverse_kinematics(current_joints_deg, T_target)
    """

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "ee_link",
        joint_names: list[str] | None = None,
    ):
        import placo

        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)

        self.target_frame_name = target_frame_name
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names

        self._tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

    # ------------------------------------------------------------------
    # FK
    # ------------------------------------------------------------------

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """Compute end-effector pose (4x4 matrix) from joint angles in degrees."""
        joint_pos_rad = np.deg2rad(joint_pos_deg[:len(self.joint_names)])
        for i, jn in enumerate(self.joint_names):
            self.robot.set_joint(jn, joint_pos_rad[i])
        self.robot.update_kinematics()
        return self.robot.get_T_world_frame(self.target_frame_name)

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        """Compute joint angles (deg) for a target 4x4 pose.

        Uses current_joint_pos as initial guess.  Iterates up to
        50× with early exit when position error < 0.1 mm.
        """
        current_joint_rad = np.deg2rad(current_joint_pos[:len(self.joint_names)])
        for i, jn in enumerate(self.joint_names):
            self.robot.set_joint(jn, current_joint_rad[i])

        self._tip_frame.T_world_frame = desired_ee_pose
        self._tip_frame.configure(self.target_frame_name, "soft",
                                  position_weight, orientation_weight)

        for _ in range(50):
            self.solver.solve(True)
            self.robot.update_kinematics()

            current_T = self.robot.get_T_world_frame(self.target_frame_name)
            error = np.linalg.norm(desired_ee_pose[:3, 3] - current_T[:3, 3])
            if error < 1e-4:
                break

        joint_pos_rad = np.array(
            [self.robot.get_joint(jn) for jn in self.joint_names]
        )
        joint_pos_deg = np.rad2deg(joint_pos_rad)

        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[:len(self.joint_names)] = joint_pos_deg
            result[len(self.joint_names):] = current_joint_pos[len(self.joint_names):]
            return result
        return joint_pos_deg
