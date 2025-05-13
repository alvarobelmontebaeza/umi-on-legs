
import torch

import rclpy
from rclpy.node import Node

import numpy as np
import os
import sys
import time

from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

from interbotix_common_modules import angle_manipulation as ang

class WX250sTestNode(Node):
    def __init__(self,
                 policy_path: str,
                 device: str = 'cpu',
                 ):
        super().__init__('wx250s_test_node')


        self.device = device
        # Load the TorchScript policy
        self.policy_path = policy_path
        self.policy = torch.jit.load(self.policy_path)
        self.policy.eval()

        # Initialize the Interbotix arm interface
        self.wx250s = InterbotixManipulatorXS(
            robot_model='wx250s',
            group_name='arm',
            gripper_name='gripper',
        )
        robot_startup()

        # Check that the arm has initialized correctly
        if (self.wx250s.arm.group_info.num_joints < 6):
            self.wx250s.get_node().logfatal(
                "The arm has not initialized correctly. Please check the connection and try again.")
            robot_shutdown()
            sys.exit()
        
        # Switch the arm to velocity profile mode
        self.wx250s.core.robot_set_operating_modes(
            cmd_type="group",
            name="arm",
            mode="position",
            profile_type="velocity",
            profile_velocity=131, # 131 == 3.14rad/s
            profile_acceleration=15, # 15 == 5.6rad/s^2
        )

        # Set PD gains for the arm
        self.wx250s.core.robot_set_motor_pid_gains(
            cmd_type="group",
            name="arm",
            kp_pos= 800,
            kd_pos= 80,
        )

        # Move the arm to the home position
        self.wx250s.arm.go_to_home_pose()
        print("Waiting for arm to reach home position...")
        time.sleep(5.0)
        ee_T = self.wx250s.arm.get_ee_pose()

        euler = ang.rotation_matrix_to_euler_angles(
            ee_T[0:3, 0:3],
            'xyz'
        )
        quat = ang.euler_angles_to_quaternion(euler)

        # Init observation variables
        self.joint_positions = torch.zeros(6, dtype=torch.float32, device=self.device)
        self.joint_velocities = torch.zeros(6, dtype=torch.float32, device=self.device)
        self.pose_command = torch.zeros(7, dtype=torch.float32, device=self.device)
        self.pose_command[0:3] = torch.tensor([0.47, 0.0, 0.33], dtype=torch.float32, device=self.device)  # x, y, z
        self.pose_command[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.last_action = torch.zeros(6, dtype=torch.float32, device=self.device)

        # Create a timer to run the control loop at 60 Hz
        print("Starting control loop...")
        self.timer = self.create_timer(1.0/60.0, self.control_loop)

    def control_loop(self):
        # Read current joint positions
        joint_positions = self.wx250s.arm.get_joint_positions()
        self.joint_positions = torch.tensor(joint_positions, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Read current joint velocities
        joint_velocities = self.wx250s.arm.get_joint_velocities()
        self.joint_velocities = torch.tensor(joint_velocities, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.obs = torch.cat((self.joint_positions, self.joint_velocities, self.pose_command.unsqueeze(0), self.last_action.unsqueeze(0)), dim=0, device=self.device)
        

        # Pass through the policy to get target joint positions
        with torch.no_grad():
            self.last_action = self.policy(self.obs)

        target_positions = self.last_action.numpy().tolist()

        # Command the arm to move to the target joint positions
        self.wx250s.arm.set_joint_positions(target_positions)