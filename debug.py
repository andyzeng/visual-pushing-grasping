#!/usr/bin/env python

import numpy as np
import time
from robot import Robot


# User options (change me)
# --------------- Setup options ---------------
tcp_host_ip = '100.127.7.223' # IP and port to robot arm as TCP client (UR5)
tcp_port = 30002
workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
# ---------------------------------------------

# Initialize robot and move to home pose
robot = Robot(False, None, None, workspace_limits,
              tcp_host_ip, tcp_port, None, None,
              False, None, None)

# Repeatedly grasp at middle of workspace
grasp_position = np.sum(workspace_limits, axis=1)/2
grasp_position[2] = -0.25

grasp_position[0] = 86 * 0.002 + workspace_limits[0][0]
grasp_position[1] = 120 * 0.002 + workspace_limits[1][0]
grasp_position[2] = workspace_limits[2][0]

while True:
    robot.grasp(grasp_position, 11*np.pi/8, workspace_limits)
    # robot.push(push_position, 0, workspace_limits)
    # robot.restart_real()
    time.sleep(1)

# Repeatedly move to workspace corners
# while True:
#     robot.move_to([workspace_limits[0][0], workspace_limits[1][0], workspace_limits[2][0]], [2.22,-2.22,0])
#     robot.move_to([workspace_limits[0][0], workspace_limits[1][1], workspace_limits[2][0]], [2.22,-2.22,0])
#     robot.move_to([workspace_limits[0][1], workspace_limits[1][1], workspace_limits[2][0]], [2.22,-2.22,0])
#     robot.move_to([workspace_limits[0][1], workspace_limits[1][0], workspace_limits[2][0]], [2.22,-2.22,0])
