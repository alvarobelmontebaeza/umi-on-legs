

import argparse
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import faulthandler

import rclpy
from rclpy.node import Node


from collections import deque
import time
import numpy as np
import os
import sys

# Interbotix SDK
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


from modules.wx250s_test import WX250sTestNode


if __name__ == "__main__":

    np.set_printoptions(precision=3)
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]"
    )
    rclpy.init(args=None)
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    wx250s_node = WX250sTestNode(**vars(args))
    logging.info("Deploy node ready")
    arm_state = wx250s_node.wx250s.arm.get_joint_positions()
    if (len(arm_state) < 6):
        logging.error("Arm is not connected!")
        exit(1)
    try:
        logging.info("Starting wx250s node...")
        rclpy.spin(wx250s_node)
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt detected. Shutting down...")
        wx250s_node.wx250s.arm.go_to_sleep_pose()
        robot_shutdown()
        rclpy.shutdown()
