
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
        self.action_scale = 0.5

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
            profile_velocity=70, # 131 == 3.14rad/s
            profile_acceleration=15, # 15 == 5.6rad/s^2
        )

        # Set PD gains for the arm
        self.wx250s.core.robot_set_motor_pid_gains(
            cmd_type="group",
            name="arm",
            kp_pos= 600,
            kd_pos= 30,
        )

        # Move the arm to the home position
        self.wx250s.arm.go_to_home_pose()
        print("Waiting for arm to reach home position...")
        time.sleep(5.0)
        ee_T = self.wx250s.arm.get_ee_pose()

        euler = ang.rotation_matrix_to_euler_angles(
            ee_T[0:3, 0:3]
        )
        quat = ang.euler_angles_to_quaternion(euler)

        # Init observation variables
        self.joint_positions = torch.zeros(6, dtype=torch.float32, device=self.device)
        self.joint_velocities = torch.zeros(6, dtype=torch.float32, device=self.device)
        self.pose_command = torch.zeros(7, dtype=torch.float32, device=self.device)
        self.pose_command[0:3] = torch.tensor(ee_T[:3,3], dtype=torch.float32, device=self.device)  # x, y, z
        self.pose_command[3:7] = torch.tensor([quat[3], quat[0], quat[1], quat[2]], dtype=torch.float32, device=self.device)
        self.last_action = torch.zeros(6, dtype=torch.float32, device=self.device)
        self.max_torque = 0.0

        # Create a timer to run the control loop at 50 Hz
        print("Starting control loop...")
        self.init_time = time.monotonic()
        self.first_update = True
        self.timer = self.create_timer(1.0/1000.0, self.control_loop)

    def control_loop(self):
        # Check current time
        current_time = time.monotonic()
        if current_time - self.init_time > 5.0:
            self.init_time = current_time
            if self.first_update:
                self.pose_command[1] += 0.15
                self.first_update = False
            else:
                self.pose_command[1] *= -1.0
                self.pose_command[2] -= 0.05

            print("pose_command: ", self.pose_command)

        # Read current joint positions
        joint_positions = self.wx250s.arm.get_joint_positions()
        self.joint_positions = torch.tensor(joint_positions, dtype=torch.float32, device=self.device).unsqueeze(0)
        print("joint_positions: ", self.joint_positions)
        # Read current joint velocities
        joint_velocities = self.wx250s.arm.get_joint_velocities()
        self.joint_velocities = torch.tensor(joint_velocities, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Read current end effector pose
        ee_T = self.wx250s.arm.get_ee_pose()
        ee_pos = torch.tensor(ee_T[:3,3], dtype=torch.float32, device=self.device).unsqueeze(0)
        ee_rot = ang.euler_angles_to_quaternion(ang.rotation_matrix_to_euler_angles(ee_T[0:3, 0:3]))
        ee_quat = torch.tensor([ee_rot[3], ee_rot[0], ee_rot[1], ee_rot[2]], dtype=torch.float32, device=self.device).unsqueeze(0)
        print("ee_pos: ", ee_pos)
        print("ee_quat: ", ee_quat)
        print("pose_command: ", self.pose_command)

        # Build the observation vector
        self.obs = torch.cat([
            self.joint_positions,
            self.joint_velocities,
            self.pose_command.unsqueeze(0),
            self.last_action.unsqueeze(0)
            ], dim=1)
        

        '''
        # Curr torque
        current = self.wx250s.core.robot_get_motor_registers("group", "arm", "Present_Current")
        torque = np.array(current[0:]) * 0.001 * 1.769
        print("torque: ", torque)

        self.max_torque = np.max(np.append(np.abs(torque), np.array(self.max_torque)))
        print("max_torque: ", self.max_torque)
        '''        

        # Pass through the policy to get target joint positions
        with torch.no_grad():
            action = self.policy(self.obs).flatten()

        target_positions = action * self.action_scale

        # Command the arm to move to the target joint positions
        #self.wx250s.arm.set_joint_positions(target_positions.numpy().tolist(), blocking=False)
        # Update the last action
        self.last_action = action
