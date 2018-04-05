import socket
import numpy as np
import cv2
import os
import time
import struct


class TCPCamera(object):
    # Written by Johnny Lee

    def __init__(self, host, port, width, height, channels, headerSize, command):
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.headerSize = headerSize
        self.channels = channels
        self.command = command
        self.bufferSize = 4098 # 4 KiB

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(1)
        self.socket.connect((self.host, self.port))

    def close(self):
        self.socket.close()

    def grabRawData(self, size):
        data = ""
        while len(data) < size:
            part = self.socket.recv(self.bufferSize)
            data += part
        return data

    def grabTimestampOnly(self):
        self.socket.send("\n")
        header = self.socket.recv(self.headerSize)
        timestamp = struct.unpack('Q', header)[0]
        return timestamp

    def grabImage(self):
        self.socket.send(self.command)
        data = self.grabRawData(self.width*self.height*self.channels)
        return np.fromstring(data, np.uint8).reshape( self.height, self.width, self.channels)


class TCPCameraRealsense(TCPCamera):

    def __init__(self, host, port):
        TCPCamera.__init__(self,host, port, 1280, 720, 1, 8, "DEPTH COLOR")

    def grabImage(self, channel):
        while True:
            self.socket.send(self.command)
            header = self.socket.recv(self.headerSize)
            payloadsize = struct.unpack('Q', header)[0]
            header = self.socket.recv(self.headerSize)
            timestamp = struct.unpack('Q', header)[0]
            data = self.grabRawData(payloadsize)
            if payloadsize == self.height*self.width*5:
                break

        if channel == 0: # Depth only
            return np.fromstring(data[:self.width*self.height*2], np.uint16).reshape(self.height, self.width)
        if channel == 1: # Color only
            return np.fromstring(data[self.width*self.height*2:], np.uint8).reshape(self.height, self.width, 3)
        if channel == 2: # Depth and color
            return np.fromstring(data[:self.width*self.height*2], np.uint16).reshape(self.height, self.width), np.fromstring(data[self.width*self.height*2:], np.uint8).reshape(self.height, self.width, 3)


class Camera(object):

    def __init__(self):

        # Data options (change me)
        self.im_height = 720
        self.im_width = 1280
        self.server_ip = '127.0.0.1'

        self.TCPCamera = TCPCameraRealsense(self.server_ip, 50000)
        self.TCPCamera.connect()


    def get_data(self):

        depth_img, color_img = self.TCPCamera.grabImage(2) 
        depth_img = depth_img.astype(float)/10000.0
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

