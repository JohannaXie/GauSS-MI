
"""
# define interest region
"""
import rospy
import torch
import numpy as np
from shapely import geometry as shapgeo
from shapely import affinity
from utils.rotation_utils import Rotation2Quaternion, Quaternion2Euler

### only 2D ###
class InterestRegion:
    def __init__(self, config):
        # interest region
        self.interest_x_min = config["Mapping"]["interest_x_min"]
        self.interest_x_max = config["Mapping"]["interest_x_max"]
        self.interest_x_len = self.interest_x_max - self.interest_x_min
        self.interest_y_min = config["Mapping"]["interest_y_min"]
        self.interest_y_max = config["Mapping"]["interest_y_max"]
        self.interest_y_len = self.interest_y_max - self.interest_y_min
        self.interest_z_min = config["Mapping"]["interest_z_min"]
        self.interest_z_max = config["Mapping"]["interest_z_max"]

        # camera detection
        self.depth_min_dist = config["Detection"]["depth_min_dist"]
        self.depth_max_dist = config["Detection"]["depth_max_dist"]
        self.camera_fov = config["Detection"]["fov"]             # degree

        # interest region
        self.interest_region = shapgeo.Polygon([(self.interest_x_min, self.interest_y_min),
                                        (self.interest_x_max, self.interest_y_min),
                                        (self.interest_x_max, self.interest_y_max),
                                        (self.interest_x_min, self.interest_y_max),
                                        (self.interest_x_min, self.interest_y_min)])
        self.init_detect_region_origin()
        self.init_gaussian_recon_region()

        # object region
        margin_xy = config["Mapping"]["object_margin_xy"]
        margin_z = config["Mapping"]["object_margin_z"]
        self.object_x_min = self.interest_x_min + margin_xy
        self.object_x_max = self.interest_x_max - margin_xy
        self.object_y_min = self.interest_y_min + margin_xy
        self.object_y_max = self.interest_y_max - margin_xy
        self.object_z_min = self.interest_z_min
        self.object_z_max = self.interest_z_max - margin_z

    # init `self.detect_region_origin` on origin point
    def init_detect_region_origin(self):
        camera_fov_2 = self.camera_fov / 2.0                # degree
        camera_fov_2_rad = np.deg2rad(camera_fov_2)    # radian

        line_base = shapgeo.LineString([(0,0), (self.depth_max_dist/np.cos(camera_fov_2_rad), 0)])      # rad
        line1 = affinity.rotate(line_base, camera_fov_2, origin=(0,0))         # degree
        line2 = affinity.rotate(line_base, -camera_fov_2, origin=(0,0))

        # detect region -- base
        self.detect_region_origin = shapgeo.Polygon([(0,0),
                                                    line1.coords[1],
                                                    line2.coords[1],
                                                    (0,0)])

    def init_gaussian_recon_region(self):
        recon_x_more = min(self.depth_max_dist, self.interest_y_len)
        self.reconstruct_x_len = self.interest_x_len + 2* recon_x_more
        recon_y_more = min(self.depth_max_dist, self.interest_x_len)
        self.reconstruct_y_len = self.interest_y_len + 2* recon_y_more

        self.reconstruct_x_min = self.interest_x_min - recon_x_more
        self.reconstruct_x_max = self.interest_x_max + recon_x_more
        self.reconstruct_y_min = self.interest_y_min - recon_y_more
        self.reconstruct_y_max = self.interest_y_max + recon_y_more

    # return the maximum depth in the reconstructed region
    # input: viewpoint position (np.array 3*1), yaw (double)
    def get_maxdepth_inregion(self, viewpoint, yaw_degree):
        if not self.in_interest_region(viewpoint[0], viewpoint[1]):
            rospy.logwarn("viewpoint not in interest region")
            return self.depth_max_dist
        
        # detect region on the viewpoint
        self.detect_region = affinity.rotate(self.detect_region_origin, yaw_degree, origin=(0,0))
        self.detect_region = affinity.translate(self.detect_region, xoff=viewpoint[0], yoff=viewpoint[1])

        max_detect_line = shapgeo.LineString([self.detect_region.exterior.coords[1], self.detect_region.exterior.coords[2]])
        interest_in_detection = self.detect_region.intersection(self.interest_region)
        max_depth4mask = self.depth_max_dist - max_detect_line.distance(interest_in_detection)
        return max_depth4mask
    
    # Get Interest Region Mask
    def get_interest_region_mask(self, camera):
        # yaw
        quat = Rotation2Quaternion(camera.R_gt_ori)
        Euler = Quaternion2Euler(quat)
        yaw_degree = Euler[2,0] * 180 / np.pi +90.0            # monogs heading y-aixs; while we heading x-axis

        # depth range
        max_depth4mask = self.get_maxdepth_inregion(camera.T_gt_ori, yaw_degree)
        min_depth4mask = self.depth_min_dist

        gt_depth = torch.from_numpy(camera.depth).to(dtype=torch.float32, device=camera.device)[None]
        max_mask = (gt_depth < max_depth4mask).squeeze()
        min_mask = (gt_depth > min_depth4mask).squeeze()
        interest_region_mask = torch.logical_and(max_mask, min_mask)
        return interest_region_mask
    
    def get_interest_region_mask_from_rawgt(self, T, depth):
        Rota = T[:3, :3]
        Translat = T[:3, 3]
        # yaw
        quat = Rotation2Quaternion(Rota)
        Euler = Quaternion2Euler(quat)
        yaw_degree = Euler[2,0] * 180 / np.pi +90.0            # monogs heading y-aixs; while we heading x-axis

        # depth range
        max_depth4mask = self.get_maxdepth_inregion(Translat, yaw_degree)
        min_depth4mask = self.depth_min_dist

        gt_depth = torch.from_numpy(depth).to(dtype=torch.float32, device="cuda")[None]
        max_mask = (gt_depth < max_depth4mask).squeeze()
        min_mask = (gt_depth > min_depth4mask).squeeze()
        interest_region_mask = torch.logical_and(max_mask, min_mask)
        return interest_region_mask
    
    def in_interest_region(self, x, y):
        return self.interest_region.contains(shapgeo.Point(x, y))
    