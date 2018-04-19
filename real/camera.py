#!/usr/bin/env python

import socket
import numpy as np
import cv2
import os
import time
import struct


class Camera(object):

    def __init__(self):

        # Data options (change me)
        self.im_height = 720
        self.im_width = 1280
        self.tcp_host_ip = '127.0.0.1'
        self.tcp_port = 50000
        self.buffer_size = 4098 # 4 KiB

        # Connect to server
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))

        self.intrinsics = None
        self.get_data()


    def get_data(self):

        # Ping the server with anything
        self.tcp_socket.send(b'asdf')

        # Fetch TCP data:
        #     color camera intrinsics, 9 floats, number of bytes: 9 x 4
        #     depth scale for converting depth from uint16 to float, 1 float, number of bytes: 4
        #     depth image, self.im_width x self.im_height uint16, number of bytes: self.im_width x self.im_height x 2
        #     color image, self.im_width x self.im_height x 3 uint8, number of bytes: self.im_width x self.im_height x 3
        data = b''
        while len(data) < (10*4 + self.im_height*self.im_width*5):
            data += self.tcp_socket.recv(self.buffer_size)

        # Reorganize TCP data into color and depth frame
        self.intrinsics = np.fromstring(data[0:(9*4)], np.float32).reshape(3, 3)
        depth_scale = np.fromstring(data[(9*4):(10*4)], np.float32)[0]
        depth_img = np.fromstring(data[(10*4):((10*4)+self.im_width*self.im_height*2)], np.uint16).reshape(self.im_height, self.im_width)
        color_img = np.fromstring(data[((10*4)+self.im_width*self.im_height*2):], np.uint8).reshape(self.im_height, self.im_width, 3)
        depth_img = depth_img.astype(float) * depth_scale
        return color_img, depth_img



# DEPRECATED CAMERA CLASS FOR REALSENSE WITH ROS
# ----------------------------------------------

# import rospy
# from realsense_camera.msg import StreamData

# class Camera(object):


#     def __init__(self):

#         # Data options (change me)
#         self.im_height = 720
#         self.im_width = 1280

#         # RGB-D data variables
#         self.color_data = np.zeros((self.im_height,self.im_width,3))
#         self.depth_data = np.zeros((self.im_height,self.im_width))
#         self.intrinsics = np.zeros((3,3))

#         # Start ROS subscriber to fetch RealSense RGB-D data
#         rospy.init_node('listener', anonymous=True)
#         rospy.Subscriber("/realsense_camera/stream", StreamData, self.realsense_stream_callback)

#         # Recording variables
#         self.frame_idx = 0
#         self.is_recording = False
#         self.recording_directory = ''

#     # ROS subscriber callback function
#     def realsense_stream_callback(self, data):
#         tmp_color_data = np.asarray(bytearray(data.color))
#         tmp_color_data.shape = (self.im_height,self.im_width,3)
#         tmp_depth_data = np.asarray(data.depth)
#         tmp_depth_data.shape = (self.im_height,self.im_width)
#         tmp_depth_data = tmp_depth_data.astype(float)/10000
#         tmp_intrinsics = np.asarray(data.intrinsics)
#         tmp_intrinsics.shape = (3,3)

#         self.color_data = tmp_color_data
#         self.depth_data = tmp_depth_data
#         self.intrinsics = tmp_intrinsics

#         if self.is_recording:
#             tmp_color_image = cv2.cvtColor(tmp_color_data, cv2.COLOR_RGB2BGR)
#             cv2.imwrite(os.path.join(self.recording_directory, '%06d.color.png' % (self.frame_idx)), tmp_color_image)
#             tmp_depth_image = np.round(tmp_depth_data * 10000).astype(np.uint16) # Save depth in 1e-4 meters
#             cv2.imwrite(os.path.join(self.recording_directory, '%06d.depth.png' % (self.frame_idx)), tmp_depth_image)
#             self.frame_idx += 1
#         else:
#             self.frame_idx = 0

#         time.sleep(0.1)

#     # Start/stop recording RGB-D video stream
#     def start_recording(self, directory):
#         self.recording_directory = directory
#         self.is_recording = True
#     def stop_recording(self):
#         self.is_recording = False

