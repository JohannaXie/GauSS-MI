import glob
import cv2
import numpy as np
import torch
from PIL import Image
import threading

import rospy
import message_filters
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge
from std_msgs.msg import Empty

from gaussian_splatting.utils.graphics_utils import focal2fov, getProjectionMatrix2
from utils.rotation_utils import Quaternion2Rot, Quaterniond

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass

"""
# for active reconstruction
"""
class RosData(BaseDataset):
    def __init__(self, args, config):
        super().__init__(args, config)
        # Camera prameters
        calibration = config["Dataset"]["Calibration"]
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
        self.projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            W=self.width,
            H=self.height,
        ).transpose(0, 1)
        self.projection_matrix = self.projection_matrix.to(device=self.device)
        self.depth_scale = calibration["depth_scale"]

        # image subscribers
        self.bgr_sub = message_filters.Subscriber('/camera/bgr', ImageMsg, queue_size=1, buff_size=2**24 * 2)
        self.depth_sub = message_filters.Subscriber('/camera/depth', ImageMsg, queue_size=1, buff_size=2**24 * 2)
        self.pose_sub = message_filters.Subscriber('/camera/pose', PoseStamped, queue_size=1, buff_size=2**24 * 2)
        self.image_ts = message_filters.TimeSynchronizer([self.bgr_sub, self.depth_sub, self.pose_sub], 10)
        self.image_ts.registerCallback(self.imageCallback)
        self.dataDone = False
        rospy.Subscriber('/grecon/end_trigger', Empty, self.dataDoneCallback)       ## external end trigger

        # messages
        self.bridge = CvBridge()
        self.thread_lock = threading.Lock()
        self.reset_image_message()

    def __getitem__(self, idx):
        self.thread_lock.acquire()

        if (type(self.rgb_image) == np.ndarray) and (type(self.depth_image) == np.ndarray) and (self.cam_ori_pose is not None):
            RGBimage = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
            RGBimage = (
                torch.from_numpy( RGBimage / 255.0)
                .clamp(0.0, 1.0)
                .permute(2, 0, 1)
                .to(device=self.device, dtype=self.dtype)
            )
            Depthimage = self.depth_image / self.depth_scale
            ori_pose = self.cam_ori_pose

            self.reset_image_message()
            self.thread_lock.release()
            return True, RGBimage, Depthimage, ori_pose

        self.reset_image_message()
        self.thread_lock.release()
        return False, None, None, None

    def dataDoneCallback(self, msg):
        self.dataDone = True

    def reset_image_message(self):
        self.rgb_image = None
        self.depth_image = None

    def getPoseMatrix(self, pos_i, quat_i):
        T = np.zeros((4,4))
        T[0:3, 0:3] = Quaternion2Rot(Quaterniond(quat_i[0], quat_i[1], quat_i[2], quat_i[3]))
        T[0:3, 3:4] = pos_i.reshape((3,1))
        T[3,3] = 1
        return T

    def imageCallback(self, rgb_msg, depth_msg, pose_msg):
        self.thread_lock.acquire()

        self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        pos = np.zeros(3)
        pos[0] = pose_msg.pose.position.x + 0.0
        pos[1] = pose_msg.pose.position.y + 0.0
        pos[2] = pose_msg.pose.position.z + 0.0
        quat = np.zeros(4)      # quaternion in x, y, z, w
        quat[0] = pose_msg.pose.orientation.x + 0.0
        quat[1] = pose_msg.pose.orientation.y + 0.0
        quat[2] = pose_msg.pose.orientation.z + 0.0
        quat[3] = pose_msg.pose.orientation.w + 0.0
        self.cam_ori_pose = self.getPoseMatrix(pos, quat)
        
        self.thread_lock.release()


def load_data(args, config):
    if config["Dataset"]["type"] == "rosbag" or config["Dataset"]["type"] == "active":
        return RosData(args, config)
    # elif config["Dataset"]["type"] == "dataset":
    #     return CollectData(args, config)
    else:
        raise ValueError("Unknown dataset type")

