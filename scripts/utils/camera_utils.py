import torch
from torch import nn
import numpy as np
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2

class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda",
        ori_T=np.zeros((4,4)),
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        # ground truth camera pose
        self.R_gt = gt_T[:3, :3]    # this is inversed
        self.T_gt = gt_T[:3, 3]
        # original rotation and translation
        self.R_gt_ori = ori_T[:3, :3]
        self.T_gt_ori = ori_T[:3, 3]

        self.R = self.R_gt.to(device=self.device)
        self.T = self.T_gt.to(device=self.device)

        # color and depth
        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        # camera intrinsics
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width
        self.projection_matrix = projection_matrix.to(device=device)

        self.interest_region_mask = torch.ones((self.image_height, self.image_width), dtype=torch.bool, device=self.device)

    @staticmethod
    def init_from_image_pose(dataset, gt_color, gt_depth, ori_pose, idx):
        gt_pose = np.linalg.inv(ori_pose)
        gt_pose = torch.from_numpy(gt_pose).to(device="cuda")
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            dataset.projection_matrix,
            dataset.fx, dataset.fy, dataset.cx, dataset.cy,
            dataset.fovx, dataset.fovy, dataset.height, dataset.width,
            device = dataset.device,
            ori_T = ori_pose,
        )

    @staticmethod
    def init_for_render(dataset):
        T=torch.from_numpy(np.eye(4)).to(device="cuda")
        return Camera(-1, None, None, T, dataset.projection_matrix,
                      dataset.fx, dataset.fy, dataset.cx, dataset.cy,
                      dataset.fovx, dataset.fovy, dataset.height, dataset.width,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H).transpose(0, 1)
        return Camera(uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W)

    @property
    def world_view_transform(self): # transformation matrix
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
        )).squeeze(0)

    @property
    def camera_center(self):    # inverse of transformation matrix
        return self.world_view_transform.inverse()[3, :3]

    def set_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    
