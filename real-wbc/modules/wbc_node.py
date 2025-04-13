import datetime
import pytz
import zarr
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
from typing import Dict, List, Optional
import typing
import logging
from rich.logging import RichHandler

from modules.common import (
    LEG_DOF,
    POS_STOP_F,
    SDK_DOF,
    VEL_STOP_F,
    MotorId,
    reorder,
    torque_limits,
)
import scipy.signal as signal
from transforms3d import affines, quaternions, euler, axangles

from modules.realtime_traj import RealtimeTraj

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crc_module import get_crc
from modules.velocity_estimator import MovingWindowFilter, VelocityEstimator
import numpy as np
import torch
import faulthandler

import rclpy
from rclpy.node import Node
from unitree_go.msg import (
    WirelessController,
    LowState,
    LowCmd,
    MotorCmd,
)
import time
import hydra
from omegaconf import OmegaConf
from geometry_msgs.msg import PoseStamped
from rclpy.time import Time

from robot_state.msg import EEFState, EEFTraj


def quat_rotate_inv(q: np.ndarray, v: np.ndarray):
    return quaternions.rotate_vector(
        v=v,
        q=quaternions.qinverse(q),
    )


from collections import deque
import time
import numpy as np
import os
import sys

from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


class WBCNode(Node):
    def __init__(
        self,
        ckpt_path: str,
        pickle_path: str,
        traj_idx: int,
        time_to_replay: float = 3.0,  # how long to wait after policy starts before starting trajectory
        replay_speed: float = 1.0,
        fix_at_init_pose: bool = False,
        use_realtime_target: bool = False,
        policy_dt_slack: float = 0.003,
        low_state_history_depth: int = 1,  # changed from 10, doesn't make much of a difference
        device: str = "cpu",
        init_pos_err_tolerance: float = 0.1,  # meters
        init_orn_err_tolerance: float = 0.5,  # radians
        logging_dir: str = "logs",
        pose_estimator: str = "mocap",
    ):
        super().__init__("deploy_node")  # type: ignore
        self.replay_speed = replay_speed
        self.time_to_replay = time_to_replay
        self.debug_log = False
        self.fix_at_init_pose = fix_at_init_pose
        self.init_action = np.zeros(18)
        self.latest_tick = -1
        
        self.prev_action = self.init_action

        # Adjust based on actual robot configuration
        self.arm2base = affines.compose(
            T=np.array([0.085, 0.0, 0.094]),
            R=np.identity(3),
            Z=np.ones(3),
        )

        '''
        # Tool center pose (tcp) in the UMI code base is different from the one in the arx5 sdk.
        # tcp is defined with z point forwards while arx5 ee pose is z pointing upwards.
        self.tcp2ee = affines.compose(
            T=np.zeros(3),
            R=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                ]
            ),
            Z=np.ones(3),
        )
        '''

        # -- INITIALIZE SUBSCRIBERS --
        self.joy_stick_sub = self.create_subscription(
            WirelessController,
            "wirelesscontroller",
            self.joy_stick_cb,
            low_state_history_depth,
        )
        self.lowlevel_state_sub = self.create_subscription(
            LowState, "lowstate", self.lowlevel_state_cb, low_state_history_depth
        )  # "/lowcmd" or  "lf/lowstate" (low frequencies)
        
        # Pose estimator for quadruped
        self.pose_estimator = pose_estimator
        if pose_estimator == "iphone":
            logging.info("Using iphone as pose estimator")
            self.robot_pose_sub = self.create_subscription(
                PoseStamped,
                "motion_estimator/robot_pose",
                self.robot_pose_cb,
                low_state_history_depth,
            )  # "/lowcmd" or  "lf/lowstate" (low frequencies)
        elif pose_estimator == "mocap":
            logging.info("Using mocap as pose estimator")
            self.robot_pose_sub = self.create_subscription(
                PoseStamped,
                "mocap/Go2Body",
                self.robot_pose_cb,
                low_state_history_depth,
            )
        elif pose_estimator == "mocap_gripper":
            logging.info("Directly using mocap on gripper")

        else:
            raise ValueError(f"Invalid pose_estimator: {pose_estimator}")
        self.robot_pose = np.identity(4, dtype=np.float32)
        self.robot_pose_tick = -1

        # Pose estimator for arm
        self.gripper_pose = np.identity(4, dtype=np.float32)
        self.gripper_pose_tick = -1
        self.gripper_pose_sub = self.create_subscription(
            PoseStamped,
            "mocap/WX250sGripper",
            self.gripper_pose_cb,
            low_state_history_depth,
        )

        # Initialize motor publisher
        self.go2_motor_pub = self.create_publisher(
            LowCmd, "lowcmd", low_state_history_depth
        )
        self.go2_cmd_msg = LowCmd()

        # init motor command
        self.motor_cmd = [
            MotorCmd(q=POS_STOP_F, dq=VEL_STOP_F, tau=0.0, kp=0.0, kd=0.0, mode=0x01)
            for _ in range(SDK_DOF)
        ]
        self.go2_cmd_msg.motor_cmd = self.motor_cmd.copy()
        self.quadruped_kp = np.zeros(12)
        self.quadruped_kd = np.zeros(12)

        # Initialize policy info
        self.policy_kp: np.ndarray
        self.policy_kd: np.ndarray
        self.policy_freq: float
        self.obs_history_len: int
        self.device = device
        self.init_policy(
            ckpt_path=ckpt_path, pickle_path=pickle_path, traj_idx=traj_idx
        )
        self.policy_dt_slack = policy_dt_slack

        # Create a quick timer for steadier timer interval
        self.policy_timer = self.create_timer(1.0 / 1000.0, self.policy_timer_callback)

        self.prev_policy_time = time.monotonic()
        self.prev_obs_time = time.monotonic()
        self.prev_obs_tick_s = -1.0
        self.prev_action_tick_s = -1.0

        self.obs = torch.zeros((self.obs_dim,), device=device)
        self._obs_history_buf = torch.zeros(
            (1, self.obs_history_len, self.obs_dim),
        ).to(device)
        self.obs_history_log: List[Dict[str, np.ndarray]] = []
        self.action_history_log: List[Dict[str, np.ndarray]] = []
        self.logging_dir = logging_dir
        self.angular_velocity_filter = MovingWindowFilter(window_size=10, data_dim=3)

        self.quadruped_dq = np.zeros(LEG_DOF)
        self.quadruped_q = np.zeros(LEG_DOF)
        self.quadruped_tau = np.zeros(LEG_DOF)

        # Joystick Callback variables
        self.start_policy = False
        self.start_policy_time = time.monotonic()
        logging.info("Press L2 to start policy")
        logging.info("Press L1 for emergent stop")
        self.key_is_pressed = False  # for key press event

        #  ------- ARM SETUP --------
        self.wx250s = InterbotixManipulatorXS(
            robot_model='wx250s',
            group_name='arm',
            gripper_name='gripper',
        )

        # Initialize arm
        robot_startup()

        # Check that initialization was successful
        if (self.wx250s.arm.group_info.num_joints < 6):
            self.wx250s.get_node().logfatal(
                'Robot initialization failed. Please check the robot and try again.'
            )
            robot_shutdown()
            sys.exit()

        # Set arm to position control and profile velocity mode
        self.wx250s.core.robot_set_operating_modes(
            cmd_type = 'group',
            name = 'arm',
            mode = 'position',
            profile_type = 'velocity',
            profile_velocity = 131, # Max velocity of 3.14 rad/s
        )
        
        # TODO: Set arm PD gains
        self.wx250s.core.robot_set_motor_pid_gains(
            cmd_type = 'group',
            name = 'arm',
            kp_pos = 5.0,
            kd_pos = 0.5,
        )

        # Send arm to home position
        self.wx250s.arm.go_to_home_pose()
        self.wx250s.gripper.release()
        self.wx250s_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.wx250s.arm.set_joint_positions(self.wx250s_cmd) # Should be same that home pose
        
        self.start_time = -1.0
        self.init_pos_err_tolerance = init_pos_err_tolerance
        self.init_orn_err_tolerance = init_orn_err_tolerance

        # Broadcast the arm state
        self.eef_state_pub = self.create_publisher(
            EEFState, "WidowGo2/eef_state", 1
        )

    def start(self):
        current_arm_state = self.wx250s.arm.get_joint_positions()
        self.init_arm_pos = current_arm_state.copy()
        self.start_time = time.monotonic()

    # obs history getters and setters
    @property
    def obs_history_buf(self) -> torch.Tensor:
        return self._obs_history_buf

    @obs_history_buf.setter
    def obs_history_buf(self, value: torch.Tensor):
        self._obs_history_buf = value

    @property
    def policy_dt(self) -> float:
        return 1.0 / self.policy_freq

    ##############################
    # subscriber callbacks
    ##############################

    # @profile
    def robot_pose_cb(self, msg):
        """
        Callback function to update the robot's pose based on incoming messages.

        Args:
            msg (PoseStamped): The incoming message containing the robot's pose information.

        Attributes:
            robot_pose (numpy.ndarray): The 4x4 transformation matrix representing the robot's pose.
            robot_pose_tick (int): The timestamp of the robot's pose in milliseconds.

        Notes:
            - The pose is composed using the position and orientation from the incoming message.
            - The timestamp is calculated differently based on the pose estimator type.
        """
        self.robot_pose = affines.compose(
            T=np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            R=quaternions.quat2mat(
                [
                    msg.pose.orientation.w,
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                ]
            ),
            Z=np.ones(3),
        )
        t = Time.from_msg(msg.header.stamp)
        if self.pose_estimator == "iphone":
            self.robot_pose_tick = int(np.rint(t.nanoseconds / 1e6))
        elif self.pose_estimator == "mocap":
            self.robot_pose_tick = int(self.prev_obs_tick_s * 1e3)

    def gripper_pose_cb(self, msg):
        """Directly using mocap to estimate gripper pose"""
        self.gripper_pose = affines.compose(
            T=np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            R=quaternions.quat2mat(
                [
                    msg.pose.orientation.w,
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                ]
            ),
            Z=np.ones(3),
        )
        t = Time.from_msg(msg.header.stamp)
        self.gripper_pose_tick = int(self.prev_obs_tick_s * 1e3)

    @property
    def ready_to_start_policy(self) -> bool:
        if (self.get_reaching_pos_err(0) > self.init_pos_err_tolerance) or (
            self.get_reaching_orn_err(0) > self.init_orn_err_tolerance
        ):
            pos_err = self.get_reaching_pos_err(0)
            orn_err = self.get_reaching_orn_err(0)
            logging.info(
                "Robot's pose is too far away from the target pose: "
                + f"pos_err: {pos_err:.03f}m, orn_err: {orn_err:.03f}rad"
            )
            return False
        return True

    def joy_stick_cb(self, msg):
        if msg.keys == 1:  # R1: start pipeline
            if not self.key_is_pressed:
                logging.info("standing up")
                self.start()
            self.key_is_pressed = True

        if msg.keys == 16:  # R2: stop policy
            if not self.key_is_pressed:
                logging.info("Stop policy")
                self.start_policy = False
        if msg.keys == 2:  # L1: emergency stop
            logging.info("Emergency stop")
            self.emergency_stop()
        if msg.keys == 32:  # L2: start policy
            if self.ready_to_start_policy:
                logging.info("Start policy")
                self.start_policy = True
                self.start_policy_time = time.monotonic()
                self.policy_ctrl_iter = 0
        # if msg.keys == int(2**15):  # Left # NOTE must map to another key, left already used in pose latency
        #     # pass

        if msg.keys == int(2**12):  # Up: reset reaching trajectory pose
            if not self.key_is_pressed:
                if (
                    self.pose_estimator in ["mocap", "iphone"]
                    and self.robot_pose_tick == -1
                    or self.pose_estimator == "mocap_gripper"
                    and self.gripper_pose_tick == -1
                ):
                    logging.info("Robot's pose is not initialized yet")
                else:
                    logging.info(
                        "Resetting reaching trajectory. "
                        f"Old reaching errors: {self.get_reaching_pos_err(0):.03f}m, {self.get_reaching_pos_err_dir(0)*1e3}, "
                        f"{self.get_reaching_orn_err(0):.01f}rad"
                    )
                    assert not self.target_relative_to_base

                    tcp_pose = self.get_obs_link_pose()
                    logging.info(f"tcp_pose: {tcp_pose}")
                    
                    if self.target_input_mode == "replay":
                        # compute a transform that would the first target reaching pose
                        # to the current tcp pose
                        target_pose_seq = np.zeros((len(self.target_pos_seq), 4, 4))
                        target_pose_seq[:, :3, :3] = self.target_rot_mat_seq
                        target_pose_seq[:, :3, 3] = self.target_pos_seq
                        target_pose_seq[:, 3, 3] = 1.0

                        transform = tcp_pose @ np.linalg.inv(target_pose_seq[0])
                        target_pose_seq = transform[None, :, :] @ target_pose_seq
                        self.target_pos_seq = target_pose_seq[:, :3, 3]
                        self.target_rot_mat_seq = target_pose_seq[:, :3, :3]
                        
                    elif self.target_input_mode == "realtime":
                        quaternions_wxyz = quaternions.mat2quat(tcp_pose[:3, :3])
                        logging.info(
                            f"{tcp_pose[:3, 3]=}, {self.prev_obs_tick_s=}, {self.latest_tick=}"
                        )
                        self.realtime_target_traj.update(
                            np.array([tcp_pose[:3, 3]]),
                            np.array([quaternions_wxyz]),
                            np.array([self.gripper_pos_cmd]),
                            np.array([self.prev_obs_tick_s + 0.1]),
                            self.prev_obs_tick_s,
                            adaptive_latency_matching=False,  # TODO: add to config
                            smoothen_time=0.1,  # TODO: add to config
                        )
                    logging.info(
                        "Reaching trajectory resetted. "
                        f"New reaching errors: {self.get_reaching_pos_err(0):.03f}m, "
                        f"{self.get_reaching_orn_err(0):.01f}rad,"
                        f"tcp pose: {tcp_pose}"
                    )
            self.key_is_pressed = True

        if msg.keys == int(2**8):  # A: Start trajectory replay in the mocap frame
            if not self.key_is_pressed:
                if (
                    self.pose_estimator in ["mocap", "iphone"]
                    and self.robot_pose_tick == -1
                    or self.pose_estimator == "mocap_gripper"
                    and self.gripper_pose_tick == -1
                ):
                    logging.info("Robot's pose is not initialized yet")
                elif self.pose_estimator == "iphone":
                    logging.info("iphone frame is not supported")
                else:
                    assert not self.target_relative_to_base

                    tcp_pose = self.tossing_0620_init_pose2
                    
                    if self.target_input_mode == "replay":
                        # compute a transform that would the first target reaching pose
                        # to the current tcp pose
                        target_pose_seq = np.zeros((len(self.target_pos_seq), 4, 4))
                        target_pose_seq[:, :3, :3] = self.target_rot_mat_seq
                        target_pose_seq[:, :3, 3] = self.target_pos_seq
                        target_pose_seq[:, 3, 3] = 1.0

                        transform = tcp_pose @ np.linalg.inv(target_pose_seq[0])
                        target_pose_seq = transform[None, :, :] @ target_pose_seq
                        self.target_pos_seq = target_pose_seq[:, :3, 3]
                        self.target_rot_mat_seq = target_pose_seq[:, :3, :3]
                    
                    logging.info(
                        "Reaching trajectory resetted. "
                        f"New reaching errors: {self.get_reaching_pos_err(0):.03f}m, "
                        f"{self.get_reaching_orn_err(0):.01f}rad,"
                        f"tcp pose: {tcp_pose}"
                    )
            self.key_is_pressed = True

        if msg.keys == int(2**9):  # B: start/stop dumping logs
            if not self.key_is_pressed:
                if self.debug_log:
                    # Dump all logs
                    self.dump_logs()
                logging.info(f"Setting debug_log to {not self.debug_log}")
                self.debug_log = not self.debug_log
            self.key_is_pressed = True

        if self.key_is_pressed:
            if msg.keys == 0:
                self.key_is_pressed = False

    # @profile
    def lowlevel_state_cb(self, msg: LowState):
        # imu data
        self.latest_tick = msg.tick
        imu_data = msg.imu_state

        # Parse IMU data
        acceleration = np.array(imu_data.accelerometer, dtype=np.float64)
        quaternion = np.array(imu_data.quaternion, dtype=np.float64)
        angular_velocity = self.angular_velocity_filter.calculate_average(
            np.array(imu_data.gyroscope, dtype=np.float64)
        )
        # Get projected gravity
        projected_gravity = quat_rotate_inv(quaternion, np.array([0, 0, -1]))
        
        # motor data
        self.quadruped_q = np.array(
            [motor_data.q for motor_data in msg.motor_state[:LEG_DOF]]
        )
        self.quadruped_dq = np.array(
            [motor_data.dq for motor_data in msg.motor_state[:LEG_DOF]]
        )
        self.quadruped_tau = np.array(
            [motor_data.tau_est for motor_data in msg.motor_state[:LEG_DOF]]
        )

        # Get foot contact force data
        foot_force = np.array(
            [msg.foot_force[foot_id] for foot_id in range(4)], dtype=np.float64
        )

        # ------ Get arm data ------       
        arm_dof_pos = self.wx250s.arm.get_joint_positions().copy()
        arm_dof_vel = self.wx250s.arm.get_joint_velocities().copy()
        arm_dof_torque = self.wx250s.arm.get_joint_efforts().copy()

        # Concatenate joint data
        dof_pos = (
            np.concatenate((reorder(self.quadruped_q), arm_dof_pos), axis=0)
        )
        dof_vel = (
            np.concatenate((reorder(self.quadruped_dq), arm_dof_vel), axis=0)
        )

        #### EEF Observation ####
        eef_state = EEFState()
        # Get arm EE pose in arm base frame
        arm_ee_transform = self.wx250s.arm.get_ee_pose()
        # Transform arm EE pose to the robot base frame
        self.arm_ee_transform = self.robot_pose @ self.arm2base @ arm_ee_transform
        

        eef_state.tick = msg.tick
        eef_state.system_time = time.monotonic()
        eef_state.eef_pose[0:3] = self.arm_ee_transform[:3, 3]
        eef_state.eef_pose[3:7] = quaternions.mat2quat(self.arm_ee_transform[:3, :3])
        eef_state.gripper_pos = self.wx250s.gripper.get_gripper_position()

        self.eef_state_pub.publish(eef_state)

        if self.pose_estimator in ["iphone", "mocap"] and self.robot_pose_tick == -1:
            return

        elif self.pose_estimator == "mocap_gripper" and self.gripper_pose_tick == -1:
            return

        # Get the target pose in robot frame
        local_target_pose = np.linalg.inv(self.robot_pose) @ global_target_pose

        # TODO: Revise criteria to validate targets
        if self.start_policy:
            if np.any(np.abs(local_target_pose[0, :3, 3]) > 0.5):
                self.emergency_stop()
            if np.any(np.abs(local_target_pose[..., :3, 3]) > 0.1):
                logging.warning(f"{local_target_pose[..., :3, 3]=} too far away")
                local_target_pose[..., :3, 3] = np.clip(
                    local_target_pose[..., :3, 3], -0.1, 0.1
                )

        # TODO: Remove observation scaling
        pos_obs = (local_target_pose[..., :3, 3]).reshape(-1)
        orn_obs = (local_target_pose[..., :2, :3]).reshape(-1)
        task_obs = np.concatenate((pos_obs, orn_obs), axis=0)


        obs = np.concatenate(state_obs + [task_obs, self.prev_action], axis=0)

        self.obs = torch.from_numpy(obs.copy()).squeeze().to(self.device, torch.float32)

        self.prev_obs_time = time.monotonic()
        self.prev_obs_tick_s = msg.tick / 1000

        if self.debug_log:
            if isinstance(target_indices, torch.Tensor):
                target_indices = target_indices.detach().cpu().numpy()

            obs_dict = {
                "quadruped_q": self.quadruped_q.copy(),
                "quadruped_dq": self.quadruped_dq.copy(),
                "quadruped_tau": self.quadruped_tau.copy(),
                "acceleration": acceleration.copy(),
                "quaternion": quaternion.copy(),
                "foot_force": foot_force.copy(),
                "angular_velocity": angular_velocity.copy(),
                "arm_dof_pos": arm_dof_pos.copy(),
                "arm_dof_vel": arm_dof_vel.copy(),
                "arm_dof_tau": arm_dof_torque.copy(),
                "dof_pos": dof_pos.copy(),
                "dof_vel": dof_vel.copy(),
                "curr_time_idx": self.curr_time_idx,
                "gripper_pos_cmd": float(self.gripper_pos_cmd),
                "target_indices": target_indices.copy(),
                "global_target_pose": global_target_pose.copy(),
                "observation_link_pose": observation_link_pose.copy(),
                "local_target_pose": local_target_pose.copy(),
                "pos_obs": pos_obs.copy(),
                "orn_obs": orn_obs.copy(),
                "task_obs": task_obs.copy(),
                "obs": obs.copy(),
                "time_since_policy_started": time.monotonic() - self.start_policy_time,
                "time_monotonic": time.monotonic(),
            }
            self.obs_history_log.append(obs_dict)

    ##############################
    # motor commands
    ##############################

    def motor_timer_callback(self):
        cb_start_time = time.monotonic()
        # send arm action
        self.wx250s.arm.set_joint_positions(self.wx250s_cmd)
        # Send legs action
        self.go2_cmd_msg.crc = get_crc(self.go2_cmd_msg)
        self.go2_motor_pub.publish(self.go2_cmd_msg)

    def set_gains(self, kp: np.ndarray, kd: np.ndarray):
        self.quadruped_kp = kp
        self.quadruped_kd = kd
        for i in range(LEG_DOF):
            self.motor_cmd[i].kp = kp[i]
            self.motor_cmd[i].kd = kd[i]

    def set_motor_position(
        self,
        q: np.ndarray,
    ):
        assert len(q) == 18
        # prepare arm action
        self.wx250s_cmd = q[12:]

        # prepare leg command
        for i in range(LEG_DOF):
            self.motor_cmd[i].q = float(q[i])
        self.go2_cmd_msg.motor_cmd = self.motor_cmd.copy()

    def emergency_stop(self):
        if self.debug_log:
            self.dump_logs()

        exit(0)

    ##############################
    # policy inference
    ##############################
    @torch.inference_mode()
    # @profile
    def policy_timer_callback(self):
        # stand up first
        stand_kp = np.ones(12) * 40.0
        stand_kd = np.ones(12) * 0.5
        stand_up_time = 5.0
        stand_up_buffer_time = 0.0

        if self.start_time == -1.0:
            return

        if not self.start_policy: # First, make the robot stand up
            time_ratio = (
                time.monotonic() - self.start_time - stand_up_buffer_time
            ) / stand_up_time
            time_ratio = max(min(1.0, time_ratio), 0.0)
            self.set_gains(kp=time_ratio * stand_kp, kd=time_ratio * stand_kd)
            action = self.prev_action * time_ratio + (1 - time_ratio) * np.zeros(18)
            wbc_action = action * self.action_scale + self.action_offset
            wbc_action[12:] = wbc_action[12:] * time_ratio + self.init_arm_pos * (
                1 - time_ratio
            )
            wbc_action[:12] = reorder(wbc_action[:12])
            gripper_pos = self.gripper_pos_cmd * time_ratio + (1 - time_ratio) * 0.0
            # send leg action
            self.set_motor_position(wbc_action, gripper_pos)
            self.motor_timer_callback()
        elif ( # When the robot is standing up, the policy starts
            time.monotonic() - self.start_policy_time
            > self.policy_dt * self.policy_ctrl_iter - self.policy_dt_slack
        ):
            self.set_gains(kp=self.policy_kp[:12], kd=self.policy_kd[:12])
            new_obs = self.obs.detach().to(self.device).clone()
            self.obs_history_buf = torch.cat(
                (self.obs_history_buf[:, 1:], new_obs[None, None, :]), dim=1
            )
            # Policy action inference
            with torch.inference_mode():
                action = self.actor(self.obs_history_buf.view(1, -1))[0]
                # print(action.cpu().numpy())
                raw_action = action
                self.prev_action = action.clone().cpu().numpy().copy()
                if self.policy_ctrl_iter % 10 == 0:
                    print(
                        f"pos err: {self.get_reaching_pos_err_dir()*1e3} mm",
                        f"orn err: {self.get_reaching_orn_err():.03f}rad",
                    )
                # Apply action scaling and offset
                wbc_action = (
                    self.prev_action.copy() * self.action_scale + self.action_offset
                )
            wbc_action[:12] = reorder(wbc_action[:12])
            self.set_motor_position(wbc_action, self.gripper_pos_cmd)
            self.motor_timer_callback()
            self.prev_policy_time = time.monotonic()
            self.prev_motor_time = time.monotonic()
            self.prev_action_tick_s = self.prev_obs_tick_s
            self.policy_ctrl_iter += 1

            if self.debug_log:
                action_dict = {
                    "policy_input": self.obs_history_buf.view(1, -1)
                    .detach()
                    .cpu()
                    .numpy(),
                    "raw_action": raw_action.detach().cpu().numpy(),
                    "clipped_action": action.detach().cpu().numpy(),
                    "reordered_wbc_action": wbc_action,
                }
                self.action_history_log.append(action_dict)
            # logging.info(f"Finish policy_timer_callback {time.monotonic() - cb_start_time:.04f}s")

    def init_policy(self, ckpt_path: str, pickle_path: str):
        logging.info("Preparing policy")
        faulthandler.enable()

        # Load config file
        config = pickle.load(
            open(os.path.join(os.path.dirname(ckpt_path), "config.pkl"), "rb")
        )

        # ------- CONFIG SETUP --------
        # Policy frequency
        self.policy_freq = 1 / (
            config["env"]["cfg"]["sim"]["dt"]
            * np.mean(config["env"]["controller"]["decimation_count"])
        )

        # Observation and action dims
        self.obs_history_len = int(config["env"]["obs_history_len"])
        print(self.obs_history_len)
        self.obs_dim = int(config["env"]["cfg"]["env"]["num_observations"])
        self.action_dim = int(config["env"]["cfg"]["env"]["num_actions"])
        placeholder_obs = torch.rand(
            self.obs_dim * self.obs_history_len,
            device=self.device,
        )

        # Action scales and offsets
        self.action_offset = (
            torch.Tensor(config["env"]["controller"]["offset"]["data"]).cpu().numpy()
        )
        self.action_scale = (
            np.ones(18) * float(config["env"]["controller"]["scale"]["fill_value"])
            if "fill_value" in config["env"]["controller"]["scale"]
            else np.array(list(config["env"]["controller"]["scale"]["data"]))
        )
        # self.obs_dof_pos_scale = float(config["env"]["state_obs"]["dof_pos"]["scale"])
        # self.obs_dof_pos_offset = np.array(
        #     config["env"]["state_obs"]["dof_pos"]["offset"]["data"]
        # )
        # self.obs_dof_vel_scale = float(config["env"]["state_obs"]["dof_vel"]["scale"])

        # ----- POLICY LOADING ------
        # load policy
        self.policy = torch.jit.load(ckpt_path, map_location=self.device)
        self.policy.eval()

        # Test loaded policy and retrieve metrics
        policy_inference_times = []
        with torch.no_grad():
            for _ in range(50):
                start = time.time()
                self.policy(
                    placeholder_obs[None, :]
                )
                policy_inference_times.append(float(time.time() - start))
        logging.info(
            f"Policy inference time: {np.mean(policy_inference_times)} ({np.std(policy_inference_times)})"
        )

        # -------- CONTROL PARAMS --------
        # init p_gains, d_gains, torque_limits, default_dof_pos_all
        self.policy_kp = np.zeros(18)
        self.policy_kd = np.zeros(18)
        # Leg gains
        self.policy_kp[:12] = float(config["env"]["controller"]["kp"]["fill_value"])
        self.policy_kd[:12] = float(config["env"]["controller"]["kd"]["fill_value"])
        # Arm gains
        self.policy_kp[12:] = float(config["env"]["controller"]["arm_kp"]["fill_value"])
        self.policy_kd[12:] = float(config["env"]["controller"]["arm_kd"]["fill_value"])

        init_pose = reorder(self.obs_dof_pos_offset) # TODO: check if this is correct
        for i in range(LEG_DOF):
            self.motor_cmd[i].q = init_pose[i]
            self.motor_cmd[i].dq = 0.0
            self.motor_cmd[i].tau = 0.0
            self.motor_cmd[i].kp = 0.0  # self.env.p_gains[i]  # 30
            self.motor_cmd[i].kd = 0.0  # float(self.env.d_gains[i])  # 0.6
        self.go2_cmd_msg.motor_cmd = self.motor_cmd.copy()

        logging.info("starting to play policy")
        logging.info(
            f"kp: {self.policy_kp}, kd: {self.policy_kd}, torque_limits: {torque_limits},"
            + f" obs_dof_pos_scale: {self.obs_dof_pos_scale}, "
            + f"obs_dof_pos_offset: {self.obs_dof_pos_offset},"
            + f" obs_dof_vel_scale: {self.obs_dof_vel_scale}, "
            + f"action_offset: {self.action_offset},"
            + f"action_scale: {self.action_scale}"
        )

        # Initialize target position and quaternion
        self.target_pos = np.zeros((3,), dtype=np.float32)
        self.target_rot = np.zeros((4,), dtype=np.float32)

        return config

    def get_reaching_pos_err(self) -> float:
        curr_target_pos = self.target_pos
        return float(np.linalg.norm(self.get_obs_link_pose()[:3, 3] - curr_target_pos))

    def get_reaching_orn_err(self) -> float:
        
        tcp_rot_mat = self.get_obs_link_pose()[:3, :3]
        rot_err_mat = curr_target_rot.T @ tcp_rot_mat
        trace = np.trace(rot_err_mat)
        # to prevent numerical instability, clip the trace to [-1, 3]
        trace = np.clip(trace, a_min=-1 + 1e-8, a_max=3 - 1e-8)
        rotation_magnitude = np.arccos((trace - 1) / 2)
        # account for symmetry
        rotation_magnitude = rotation_magnitude % (2 * np.pi)
        rotation_magnitude = min(
            rotation_magnitude,
            2 * np.pi - rotation_magnitude,
        )
        return rotation_magnitude

    def get_tcp_pose(self) -> np.ndarray:
        """
        In task frame
        """ 
        # Obtain the current arm end effector pose
        # in the arm base frame
        wx250s_ee_pose = self.wx250s.arm.get_ee_pose()
        # Convert from numpy array to affine transformation
        ee2arm = affines.compose(
            T=wx250s_ee_pose[:3, 3], R=wx250s_ee_pose[:3, :3], Z=np.ones(3)
        )
        # Return the end effector pose in the task frame
        return self.robot_pose @ self.arm2base @ ee2arm

    def get_obs_link_pose(self) -> np.ndarray:
        if self.pose_estimator in ["iphone", "mocap"]:
            return self.get_tcp_pose()
        elif self.pose_estimator == "mocap_gripper":
            return self.gripper_pose

    def dump_logs(self):
        obs_history_log = self.obs_history_log
        action_history_log = self.action_history_log
        timezone = pytz.timezone("Europe/Madrid")
        timestamp = datetime.datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
        logging.info(f"Dumping logs to {self.logging_dir}/{timestamp}")
        dump_start_time = time.monotonic()
        np.save(
            f"{self.logging_dir}/{timestamp}_obs_history.npy",
            obs_history_log,
            allow_pickle=True,
        )
        np.save(
            f"{self.logging_dir}/{timestamp}_action_history.npy",
            action_history_log,
            allow_pickle=True,
        )
        logging.info(f"Logs dumped, time spent: {time.monotonic() - dump_start_time}")

        self.obs_history_log = []
        self.action_history_log = []