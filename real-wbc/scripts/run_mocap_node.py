import rclpy
import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.mocap.mocap_node import MocapNode
import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str)
    args = parser.parse_args()
    rclpy.init()
    agent = MocapNode(rigid_body_dict={1:"Go2Body", 2:"WX250sGripper"}, ip=args.ip)

    try:
        rclpy.spin(agent)
    finally:
        rclpy.shutdown()
