#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import time
import cv2
import os
import datetime
import random
from collections import OrderedDict
from collections import namedtuple
from camera import Camera
from robot import Robot
import torch  
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms


# User options (change me)
# --------------- Setup options ---------------
tcp_host_ip = '100.127.7.223' # IP and port to robot arm as TCP client (UR5)
tcp_port = 30002
rtc_host_ip = '100.127.7.223' # IP and port to robot arm as real-time client (UR5)
rtc_port = 30003
workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)

tool_orientation = [2.22,-2.22,0]

# Move robot to home pose
robot = Robot(False, None, None, workspace_limits,
              tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
              False, None, None)

# print()

grasp_position = np.sum(workspace_limits, axis=1)/2
grasp_position[2] = -0.25
while True:
    robot.grasp(grasp_position, 0, workspace_limits)
    time.sleep(1)

# while True:
#     robot.restart_real()


# push_position = np.sum(workspace_limits, axis=1)/2
# push_position[2] = -0.15

# push_position = [0.49,0.11,-0.15]

# while True:
#     robot.push(push_position, -np.pi/2, workspace_limits)
#     time.sleep(1)




# # Trace perimeter of workspace until guard
# while True:
#     robot.move_to([workspace_limits[0][0], workspace_limits[1][0], workspace_limits[2][0]], tool_orientation)
#     robot.move_to([workspace_limits[0][0], workspace_limits[1][1], workspace_limits[2][0]], tool_orientation)
#     robot.move_to([workspace_limits[0][1], workspace_limits[1][1], workspace_limits[2][0]], tool_orientation)
#     robot.move_to([workspace_limits[0][1], workspace_limits[1][0], workspace_limits[2][0]], tool_orientation)
    
#     # execute_success = robot.guarded_move_to([workspace_limits[0][0], workspace_limits[1][0], workspace_limits[2][0]], tool_orientation)
#     # if not execute_success:
#     #     exit()
#     # execute_success = robot.guarded_move_to([workspace_limits[0][0], workspace_limits[1][1], workspace_limits[2][0]], tool_orientation)
#     # if not execute_success:
#     #     exit()
#     # execute_success = robot.guarded_move_to([workspace_limits[0][1], workspace_limits[1][1], workspace_limits[2][0]], tool_orientation)
#     # if not execute_success:
#     #     exit()
#     # execute_success = robot.guarded_move_to([workspace_limits[0][1], workspace_limits[1][0], workspace_limits[2][0]], tool_orientation)
#     # if not execute_success:
#     #     exit()















