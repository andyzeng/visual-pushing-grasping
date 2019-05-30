import struct
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_pointcloud(color_img, depth_img, camera_intrinsics):

    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]

    # Project depth into 3D point cloud in camera coordinates
    pix_x,pix_y = np.meshgrid(np.linspace(0,im_w-1,im_w), np.linspace(0,im_h-1,im_h))
    cam_pts_x = np.multiply(pix_x-camera_intrinsics[0][2],depth_img/camera_intrinsics[0][0])
    cam_pts_y = np.multiply(pix_y-camera_intrinsics[1][2],depth_img/camera_intrinsics[1][1])
    cam_pts_z = depth_img.copy()
    cam_pts_x.shape = (im_h*im_w,1)
    cam_pts_y.shape = (im_h*im_w,1)
    cam_pts_z.shape = (im_h*im_w,1)

    # Reshape image into colors for 3D point cloud
    rgb_pts_r = color_img[:,:,0]
    rgb_pts_g = color_img[:,:,1]
    rgb_pts_b = color_img[:,:,2]
    rgb_pts_r.shape = (im_h*im_w,1)
    rgb_pts_g.shape = (im_h*im_w,1)
    rgb_pts_b.shape = (im_h*im_w,1)

    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

    return cam_pts, rgb_pts


def get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution):

    # Compute heightmap size
    heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution, (workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution)).astype(int)

    # Get 3D point cloud from RGB-D images
    surface_pts, color_pts = get_pointcloud(color_img, depth_img, cam_intrinsics)

    # Transform 3D point cloud from camera coordinates to robot coordinates
    surface_pts = np.transpose(np.dot(cam_pose[0:3,0:3],np.transpose(surface_pts)) + np.tile(cam_pose[0:3,3:],(1,surface_pts.shape[0])))

    # Sort surface points by z value
    sort_z_ind = np.argsort(surface_pts[:,2])
    surface_pts = surface_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]

    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(np.logical_and(surface_pts[:,0] >= workspace_limits[0][0], surface_pts[:,0] < workspace_limits[0][1]), surface_pts[:,1] >= workspace_limits[1][0]), surface_pts[:,1] < workspace_limits[1][1]), surface_pts[:,2] < workspace_limits[2][1])
    surface_pts = surface_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]

    # Create orthographic top-down-view RGB-D heightmaps
    color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:,0] - workspace_limits[0][0])/heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:,1] - workspace_limits[1][0])/heightmap_resolution).astype(int)
    color_heightmap_r[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[0]]
    color_heightmap_g[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[1]]
    color_heightmap_b[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[2]]
    color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = surface_pts[:,2]
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan

    return color_heightmap, depth_heightmap

# Save a 3D point cloud to a binary .ply file
def pcwrite(xyz_pts, filename, rgb_pts=None):
    assert xyz_pts.shape[1] == 3, 'input XYZ points should be an Nx3 matrix'
    if rgb_pts is None:
        rgb_pts = np.ones(xyz_pts.shape).astype(np.uint8)*255
    assert xyz_pts.shape == rgb_pts.shape, 'input RGB colors should be Nx3 matrix and same size as input XYZ points'

    # Write header for .ply file
    pc_file = open(filename, 'wb')
    pc_file.write(bytearray('ply\n', 'utf8'))
    pc_file.write(bytearray('format binary_little_endian 1.0\n', 'utf8'))
    pc_file.write(bytearray(('element vertex %d\n' % xyz_pts.shape[0]), 'utf8'))
    pc_file.write(bytearray('property float x\n', 'utf8'))
    pc_file.write(bytearray('property float y\n', 'utf8'))
    pc_file.write(bytearray('property float z\n', 'utf8'))
    pc_file.write(bytearray('property uchar red\n', 'utf8'))
    pc_file.write(bytearray('property uchar green\n', 'utf8'))
    pc_file.write(bytearray('property uchar blue\n', 'utf8'))
    pc_file.write(bytearray('end_header\n', 'utf8'))

    # Write 3D points to .ply file
    for i in range(xyz_pts.shape[0]):
        pc_file.write(bytearray(struct.pack("fffccc",xyz_pts[i][0],xyz_pts[i][1],xyz_pts[i][2],rgb_pts[i][0].tostring(),rgb_pts[i][1].tostring(),rgb_pts[i][2].tostring())))
    pc_file.close()


def get_affordance_vis(grasp_affordances, input_images, num_rotations, best_pix_ind):
    vis = None
    for vis_row in range(num_rotations/4):
        tmp_row_vis = None
        for vis_col in range(4):
            rotate_idx = vis_row*4+vis_col
            affordance_vis = grasp_affordances[rotate_idx,:,:]
            affordance_vis[affordance_vis < 0] = 0 # assume probability
            # affordance_vis = np.divide(affordance_vis, np.max(affordance_vis))
            affordance_vis[affordance_vis > 1] = 1 # assume probability
            affordance_vis.shape = (grasp_affordances.shape[1], grasp_affordances.shape[2])
            affordance_vis = cv2.applyColorMap((affordance_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
            input_image_vis = (input_images[rotate_idx,:,:,:]*255).astype(np.uint8)
            input_image_vis = cv2.resize(input_image_vis, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            affordance_vis = (0.5*cv2.cvtColor(input_image_vis, cv2.COLOR_RGB2BGR) + 0.5*affordance_vis).astype(np.uint8)
            if rotate_idx == best_pix_ind[0]:
                affordance_vis = cv2.circle(affordance_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
            if tmp_row_vis is None:
                tmp_row_vis = affordance_vis
            else:
                tmp_row_vis = np.concatenate((tmp_row_vis,affordance_vis), axis=1)
        if vis is None:
            vis = tmp_row_vis
        else:
            vis = np.concatenate((vis,tmp_row_vis), axis=0)

    return vis


def get_difference(color_heightmap, color_space, bg_color_heightmap):

    color_space = np.concatenate((color_space, np.asarray([[0.0, 0.0, 0.0]])), axis=0)
    color_space.shape = (color_space.shape[0], 1, 1, color_space.shape[1])
    color_space = np.tile(color_space, (1, color_heightmap.shape[0], color_heightmap.shape[1], 1))

    # Normalize color heightmaps
    color_heightmap = color_heightmap.astype(float)/255.0
    color_heightmap.shape = (1, color_heightmap.shape[0], color_heightmap.shape[1], color_heightmap.shape[2])
    color_heightmap = np.tile(color_heightmap, (color_space.shape[0], 1, 1, 1))

    bg_color_heightmap = bg_color_heightmap.astype(float)/255.0
    bg_color_heightmap.shape = (1, bg_color_heightmap.shape[0], bg_color_heightmap.shape[1], bg_color_heightmap.shape[2])
    bg_color_heightmap = np.tile(bg_color_heightmap, (color_space.shape[0], 1, 1, 1))

    # Compute nearest neighbor distances to key colors
    key_color_dist = np.sqrt(np.sum(np.power(color_heightmap - color_space,2), axis=3))
    # key_color_dist_prob = F.softmax(Variable(torch.from_numpy(key_color_dist), volatile=True), dim=0).data.numpy()

    bg_key_color_dist = np.sqrt(np.sum(np.power(bg_color_heightmap - color_space,2), axis=3))
    # bg_key_color_dist_prob = F.softmax(Variable(torch.from_numpy(bg_key_color_dist), volatile=True), dim=0).data.numpy()

    key_color_match = np.argmin(key_color_dist, axis=0)
    bg_key_color_match = np.argmin(bg_key_color_dist, axis=0)
    key_color_match[key_color_match == color_space.shape[0] - 1] = color_space.shape[0] + 1
    bg_key_color_match[bg_key_color_match == color_space.shape[0] - 1] = color_space.shape[0] + 2

    return np.sum(key_color_match == bg_key_color_match).astype(float)/np.sum(bg_key_color_match < color_space.shape[0]).astype(float)


# Get rotation matrix from euler angles
def euler2rotm(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])         
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])            
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


# Checks if a matrix is a valid rotation matrix.
def isRotm(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
def rotm2euler(R) :
 
    assert(isRotm(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


def angle2rotm(angle, axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis/np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array([[ 0.0,     -axis[2],  axis[1]],
                      [ axis[2], 0.0,      -axis[0]],
                      [-axis[1], axis[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:

        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def rotm2angle(R):
    # From: euclideanspace.com

    epsilon = 0.01 # Margin to allow for rounding errors
    epsilon2 = 0.1 # Margin to distinguish between 0 and 180 degrees

    assert(isRotm(R))

    if ((abs(R[0][1]-R[1][0])< epsilon) and (abs(R[0][2]-R[2][0])< epsilon) and (abs(R[1][2]-R[2][1])< epsilon)):
        # Singularity found
        # First check for identity matrix which must have +1 for all terms in leading diagonaland zero in other terms
        if ((abs(R[0][1]+R[1][0]) < epsilon2) and (abs(R[0][2]+R[2][0]) < epsilon2) and (abs(R[1][2]+R[2][1]) < epsilon2) and (abs(R[0][0]+R[1][1]+R[2][2]-3) < epsilon2)):
            # this singularity is identity matrix so angle = 0
            return [0,1,0,0] # zero angle, arbitrary axis

        # Otherwise this singularity is angle = 180
        angle = np.pi
        xx = (R[0][0]+1)/2
        yy = (R[1][1]+1)/2
        zz = (R[2][2]+1)/2
        xy = (R[0][1]+R[1][0])/4
        xz = (R[0][2]+R[2][0])/4
        yz = (R[1][2]+R[2][1])/4
        if ((xx > yy) and (xx > zz)): # R[0][0] is the largest diagonal term
            if (xx< epsilon):
                x = 0
                y = 0.7071
                z = 0.7071
            else:
                x = np.sqrt(xx)
                y = xy/x
                z = xz/x
        elif (yy > zz): # R[1][1] is the largest diagonal term
            if (yy< epsilon):
                x = 0.7071
                y = 0
                z = 0.7071
            else:
                y = np.sqrt(yy)
                x = xy/y
                z = yz/y
        else: # R[2][2] is the largest diagonal term so base result on this
            if (zz< epsilon):
                x = 0.7071
                y = 0.7071
                z = 0
            else:
                z = np.sqrt(zz)
                x = xz/z
                y = yz/z
        return [angle,x,y,z] # Return 180 deg rotation

    # As we have reached here there are no singularities so we can handle normally
    s = np.sqrt((R[2][1] - R[1][2])*(R[2][1] - R[1][2]) + (R[0][2] - R[2][0])*(R[0][2] - R[2][0]) + (R[1][0] - R[0][1])*(R[1][0] - R[0][1])) # used to normalise
    if (abs(s) < 0.001):
        s = 1 

    # Prevent divide by zero, should not happen if matrix is orthogonal and should be
    # Caught by singularity test above, but I've left it in just in case
    angle = np.arccos(( R[0][0] + R[1][1] + R[2][2] - 1)/2)
    x = (R[2][1] - R[1][2])/s
    y = (R[0][2] - R[2][0])/s
    z = (R[1][0] - R[0][1])/s
    return [angle,x,y,z]


# Cross entropy loss for 2D outputs
class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)















