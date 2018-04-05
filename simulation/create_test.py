#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import time
import cv2
import os
import random
from collections import namedtuple
import torch  
from torch.autograd import Variable
from robot import Robot
from trainer import Trainer
from logger import Logger
import utils








# User options (change me)
is_sim = False # Run in simulation?
sim_params = {'obj_mesh_dir' : '/home/andyz/persistence/robot/meshes/blocks', # Directory containing 3D mesh files (.obj) of objects to be added to simulation
              'num_obj'      : 10} # Number of objects to add to simulation
real_params = {'tcp_host_ip' : '100.127.7.223', #'192.168.1.100', # IP and port to robot arm as TCP client (UR5)
               'tcp_port'    : 30002, 
               'rtc_host_ip' : '100.127.7.223', #'192.168.1.100', # IP and port to robot arm as real-time client (UR5)
               'rtc_port'    : 30003}
workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
heightmap_resolution = 0.002 # Meters per pixel
random_seed = 1234
method = 'reactive' # 'reactive' or 'reinforcement'
# is_testing = False
# test_file = None
is_testing = True
test_file = 'test-10-obj-10.txt'

# TCP position = 0.214m

# Set random seed
np.random.seed(random_seed)

# Initialize pick-and-place system (camera and robot)
robot = Robot(is_sim, sim_params, real_params, workspace_limits, is_testing, test_file)






test_case_file_name = raw_input("Enter the name of the file: ") # test-10-obj-00.txt

obj_positions, obj_orientations = robot.get_obj_positions_and_orientations()

file = open(test_case_file_name, 'w') 
for object_idx in range(robot.num_obj):
    curr_mesh_file = os.path.join(robot.obj_mesh_dir, robot.mesh_list[robot.obj_mesh_ind[object_idx]])
    file.write('%s %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e\n' % (curr_mesh_file,
                                                                               robot.obj_mesh_color[object_idx][0], robot.obj_mesh_color[object_idx][1], robot.obj_mesh_color[object_idx][2],
                                                                               obj_positions[object_idx][0], obj_positions[object_idx][1], obj_positions[object_idx][2],
                                                                               obj_orientations[object_idx][0], obj_orientations[object_idx][1], obj_orientations[object_idx][2]))
file.close()



# print(robot.obj_mesh_color)

# self.num_obj

# for object_idx in range(len(self.obj_mesh_ind)):
#     curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])

# save object positions to file

# print(obj_positions)
# print(obj_orientations)





exit()


