import time
import datetime
import os
import numpy as np
import cv2
import torch 
# import h5py 

class Logger():

    def __init__(self, continue_logging, logging_directory):

        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        self.continue_logging = continue_logging
        if self.continue_logging:
            self.base_directory = logging_directory
            print('Pre-loading data logging session: %s' % (self.base_directory))
        else:
            self.base_directory = os.path.join(logging_directory, timestamp_value.strftime('%Y-%m-%d.%H:%M:%S'))
            print('Creating data logging session: %s' % (self.base_directory))
        self.info_directory = os.path.join(self.base_directory, 'info')
        self.color_images_directory = os.path.join(self.base_directory, 'data', 'color-images')
        self.depth_images_directory = os.path.join(self.base_directory, 'data', 'depth-images')
        self.color_heightmaps_directory = os.path.join(self.base_directory, 'data', 'color-heightmaps')
        self.depth_heightmaps_directory = os.path.join(self.base_directory, 'data', 'depth-heightmaps')
        self.models_directory = os.path.join(self.base_directory, 'models')
        self.visualizations_directory = os.path.join(self.base_directory, 'visualizations')
        self.recordings_directory = os.path.join(self.base_directory, 'recordings')
        self.transitions_directory = os.path.join(self.base_directory, 'transitions')

        if not os.path.exists(self.info_directory):
            os.makedirs(self.info_directory)
        if not os.path.exists(self.color_images_directory):
            os.makedirs(self.color_images_directory)
        if not os.path.exists(self.depth_images_directory):
            os.makedirs(self.depth_images_directory)
        if not os.path.exists(self.color_heightmaps_directory):
            os.makedirs(self.color_heightmaps_directory)
        if not os.path.exists(self.depth_heightmaps_directory):
            os.makedirs(self.depth_heightmaps_directory)
        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)
        if not os.path.exists(self.visualizations_directory):
            os.makedirs(self.visualizations_directory)
        if not os.path.exists(self.recordings_directory):
            os.makedirs(self.recordings_directory)
        if not os.path.exists(self.transitions_directory):
            os.makedirs(os.path.join(self.transitions_directory, 'data'))

    def save_camera_info(self, intrinsics, pose, depth_scale):
        np.savetxt(os.path.join(self.info_directory, 'camera-intrinsics.txt'), intrinsics, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'camera-pose.txt'), pose, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'camera-depth-scale.txt'), [depth_scale], delimiter=' ')

    def save_heightmap_info(self, boundaries, resolution):
        np.savetxt(os.path.join(self.info_directory, 'heightmap-boundaries.txt'), boundaries, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'heightmap-resolution.txt'), [resolution], delimiter=' ')

    def save_images(self, iteration, color_image, depth_image, mode):
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.color_images_directory, '%06d.%s.color.png' % (iteration, mode)), color_image)
        depth_image = np.round(depth_image * 10000).astype(np.uint16) # Save depth in 1e-4 meters
        cv2.imwrite(os.path.join(self.depth_images_directory, '%06d.%s.depth.png' % (iteration, mode)), depth_image)
    
    def save_heightmaps(self, iteration, color_heightmap, depth_heightmap, mode):
        color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.color_heightmaps_directory, '%06d.%s.color.png' % (iteration, mode)), color_heightmap)
        depth_heightmap = np.round(depth_heightmap * 100000).astype(np.uint16) # Save depth in 1e-5 meters
        cv2.imwrite(os.path.join(self.depth_heightmaps_directory, '%06d.%s.depth.png' % (iteration, mode)), depth_heightmap)
    
    def write_to_log(self, log_name, log):
        np.savetxt(os.path.join(self.transitions_directory, '%s.log.txt' % log_name), log, delimiter=' ')

    def save_model(self, iteration, model, name):
        torch.save(model.cpu().state_dict(), os.path.join(self.models_directory, 'snapshot-%06d.%s.pth' % (iteration, name)))

    def save_backup_model(self, model, name):
        torch.save(model.state_dict(), os.path.join(self.models_directory, 'snapshot-backup.%s.pth' % (name)))

    def save_visualizations(self, iteration, affordance_vis, name):
        cv2.imwrite(os.path.join(self.visualizations_directory, '%06d.%s.png' % (iteration,name)), affordance_vis)

    # def save_state_features(self, iteration, state_feat):
    #     h5f = h5py.File(os.path.join(self.visualizations_directory, '%06d.state.h5' % (iteration)), 'w')
    #     h5f.create_dataset('state', data=state_feat.cpu().data.numpy())
    #     h5f.close()

    # Record RGB-D video while executing primitive
    # recording_directory = logger.make_new_recording_directory(iteration)
    # camera.start_recording(recording_directory)
    # camera.stop_recording()
    def make_new_recording_directory(self, iteration):
        recording_directory = os.path.join(self.recordings_directory, '%06d' % (iteration))
        if not os.path.exists(recording_directory):
            os.makedirs(recording_directory)
        return recording_directory

    def save_transition(self, iteration, transition):
        depth_heightmap = np.round(transition.state * 100000).astype(np.uint16) # Save depth in 1e-5 meters
        cv2.imwrite(os.path.join(self.transitions_directory, 'data', '%06d.0.depth.png' % (iteration)), depth_heightmap)
        next_depth_heightmap = np.round(transition.next_state * 100000).astype(np.uint16) # Save depth in 1e-5 meters
        cv2.imwrite(os.path.join(self.transitions_directory, 'data', '%06d.1.depth.png' % (iteration)), next_depth_heightmap)
        # np.savetxt(os.path.join(self.transitions_directory, '%06d.action.txt' % (iteration)), [1 if (transition.action == 'grasp') else 0], delimiter=' ')
        # np.savetxt(os.path.join(self.transitions_directory, '%06d.reward.txt' % (iteration)), [reward_value], delimiter=' ')
