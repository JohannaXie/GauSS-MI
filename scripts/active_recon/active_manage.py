"""
# Description: FSM Manager for Active Reconstruction
# Updated: Jun. 25, 2025
# Author: Yuhan (Johanna) Xie
# Email: yuhanxie@connect.hku.hk
"""

import rospy
import numpy as np
from std_msgs.msg import Empty
from geometry_msgs.msg import PoseStamped

from utils.logging_utils import Log
from utils.rotation_utils import Euler2Quaternion, Euler2Rot, rotConvert

import torch
from munch import munchify
import threading
from utils.multiprocessing_utils import clone_obj
from utils.camera_utils import Camera
from gaussian_splatting.gaussian_renderer import render

class ActiveReconFSM:
    def __init__(self, config, dataconfig):
        """ # Initializations """
        self.init_parameters(dataconfig)
        self.active_state = "init"      # FSM state

        """ # for view planning """
        self.previous_vps = []          # all previous viewpoints # in world frame, [pos, yaw(rad)]
        self.view_action_lib = []       # (dy, dz, dyaw), deg
        self.next_view_lib = []         # in world frame, [pos, yaw(rad)]
        self.frame_cnts = 0
        self.take_picture = 0

        # image-pose receiver
        self.view_camera_stack = []

        """ # for recon done check """
        self.thread_lock_gs = threading.Lock()             # lock gaussians updating in FSM
        
        # 3D Gaussian Render
        self.gaussians = None
        self.gaussians_frontend = None
        self.gaussians_inited = False
        # render parameters
        self.pipeline_params = munchify(config["pipeline_params"])
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        # viewpoint camera for rendering
        self.viewpoint_cam = None
        self.data_reader = None
        # check recon done
        self.recon_done = False

        """ # ROS Initializations """
        # ROS Timer - main FSM
        rospy.Timer(rospy.Duration(self.planning_time), self.RobotTimer)
        rospy.Timer(rospy.Duration(1.0/self.fsm_freq), self.ActiveReconFSMTimer)       ### check frequency

        # Publisher
        self.recon_done_pub = rospy.Publisher('/recon/recon_done', Empty, queue_size=150)
        self.set_pose_pub = rospy.Publisher('/gaussmi/nbv_pose', PoseStamped, queue_size=1)

        self.init_view_lib()

        """ ### Robot Implementation ###
        # Robot Initialization
        """
        self.robot_running = True

    def init_parameters(self, dataconfig):
        self.data_type = dataconfig["Dataset"]["type"]
        self.fsm_freq = dataconfig["ViewPlan"]["fsm_frequency"]
        self.device = "cuda:0"
        self.dtype = torch.float32

        # view plan parameters
        start_x = dataconfig["ViewPlan"]["start_x"]
        start_y = dataconfig["ViewPlan"]["start_y"]
        start_z = dataconfig["ViewPlan"]["start_z"]
        self.start_pos = np.array([[start_x], [start_y], [start_z]])
        self.start_yaw_deg = dataconfig["ViewPlan"]["start_yaw_deg"]

        self.planning_time = dataconfig["ViewPlan"]["plan_time"]
        self.y_action_unit = dataconfig["ViewPlan"]["y_action_unit"]
        self.z_action_unit = dataconfig["ViewPlan"]["z_action_unit"]
        self.yaw_action_unit = dataconfig["ViewPlan"]["yaw_action_unit_deg"]

        # robot workspace       # in world frame, [min, max]
        self.work_region = np.zeros([2, 3])
        self.work_region[0, 0] = dataconfig["Mapping"]["interest_x_min"]
        self.work_region[1, 0] = dataconfig["Mapping"]["interest_x_max"]
        self.work_region[0, 1] = dataconfig["Mapping"]["interest_y_min"] 
        self.work_region[1, 1] = dataconfig["Mapping"]["interest_y_max"]
        self.work_region[0, 2] = dataconfig["Mapping"]["interest_z_min"]
        self.work_region[1, 2] = dataconfig["Mapping"]["interest_z_max"]

        # previous check
        self.max_num_previous_vps = dataconfig["ViewPlan"]["max_num_previous_vps"]
        self.pre_max_pos_dist = dataconfig["ViewPlan"]["pre_max_pos_dist"]
        self.pre_max_yaw_diff = dataconfig["ViewPlan"]["pre_max_yaw_diff"] / 180.0 * np.pi
        self.pre_discount_coeff = dataconfig["ViewPlan"]["pre_discount_coeff"]

        # done check
        self.done_percent_smooth = 0
        self.done_alpha = dataconfig["ViewPlan"]["done_alpha"]
        self.done_prob = dataconfig["ViewPlan"]["done_prob"]
        self.done_threshold = dataconfig["ViewPlan"]["done_threshold"]

    def init_view_lib(self):
        """
        # in the world frame:
        # +z is up
        # +x is forward
        # +y is left
        """

        """ # initial camera poses for online 3dgs """
        self.append_viewpoint(
            self.start_pos, 
            self.start_yaw_deg / 180.0 * np.pi
        )
        self.append_viewpoint(
            self.start_pos + np.array([[0.0],[self.y_action_unit], [0.0]]), 
            (self.start_yaw_deg - self.yaw_action_unit) / 180.0 * np.pi
        )
        self.append_viewpoint(
            self.start_pos + np.array([[0.0], [0.0], [self.z_action_unit]]), 
            self.start_yaw_deg / 180.0 * np.pi
        )
            
        """ # init view-action library """
        # Sample 1: basic sample, 5x3x3 -3 = 42
        for y_sample in [-2, -1, 0, 1, 2]:
            for z_sample in [-1, 0, 1]:
                if y_sample == 0 and z_sample == 0:
                    continue
                for yaw_sample in [-1, 0, 1]:
                    self.view_action_lib.append(
                        [y_sample * self.y_action_unit, 
                         z_sample * self.z_action_unit, 
                         yaw_sample * self.yaw_action_unit])

        # Sample 2: high z sample, 4x3 = 12
        for z_sample in [-3, -2, 2, 3]:
            for yaw_sample in [-1, 0, 1]:
                self.view_action_lib.append(
                    [0.0, 
                     z_sample * self.z_action_unit, 
                     yaw_sample * self.yaw_action_unit])

        # Sample 3: high yaw sample, 6x4 = 24
        for y_sample in [-3, -2, -1, 1, 2, 3]:
            for yaw_sample in [-3, -2, 2, 3]:
                self.view_action_lib.append(
                    [y_sample * self.y_action_unit, 
                     0.0,
                     yaw_sample * self.yaw_action_unit])
                
        # Sample 4: agressive sample: 4x2x2 = 8
        for y_sample in [-4, -3, 3, 4]:
            for z_sample in [-3, 3]:
                for yaw_sample in [-4, 4]:
                    self.view_action_lib.append(
                        [y_sample * self.y_action_unit, 
                         z_sample * self.z_action_unit, 
                         yaw_sample * self.yaw_action_unit])
        
        Log(f"View Action Library Initialized. Total {len(self.view_action_lib)} samples.", tag="ActiveManage")
        
    def init_data_reader(self, datareader):
        self.data_reader = datareader
        self.viewpoint_cam = Camera.init_for_render(self.data_reader)

    """
    # Main FSM Timer:
    # NBV sampling and selecting
    """
    def ActiveReconFSMTimer(self, event):
        if self.recon_done:
            return
        
        if self.active_state == "init":
            # receive initial images data for online 3dgs
            if self.take_picture <= 0:
                if self.gaussians_inited and self.viewpoint_cam is not None:
                    Log('... Active Recon Initialized. ', tag="ActiveManage")
                    self.active_state = "nbv_plan"
                    # Log(f"Active Recon State ==> nbv_plan ...", tag="ActiveManage")
                return
        
        elif self.active_state == "nbv_plan":
            self.viewpoint_sample()
            nbv_pos, nbv_yaw_rad = self.nbv_select()
            self.append_viewpoint(nbv_pos, nbv_yaw_rad)
            self.active_state = "robot_control"
            # Log(f"Active Recon State ==> robot_control ...", tag="ActiveManage")

        elif self.active_state == "robot_control":
            ### wait for (1) robot motion to the set pose, (2) image taken ###
            if self.take_picture <= 0:
                self.active_state = "nbv_plan"
                # Log(f"Active Recon State ==> nbv_plan ...", tag="ActiveManage")

    def viewpoint_sample(self):
        [last_pos, last_yaw_rad] = self.previous_vps[-1]
        self.next_view_lib = []                             # in world frame, [pos, yaw(rad)]

        for action in self.view_action_lib:                 # (dy, dz, dyaw) deg
            next_yaw_rad = last_yaw_rad + action[2] /180.0*np.pi
            pos_step = np.array([[-action[0] * np.sin(next_yaw_rad)], 
                                 [ action[0] * np.cos(next_yaw_rad)], 
                                 [ action[1]]])
            next_pos = last_pos + pos_step
            if not self.isValid(next_pos):
                continue
            self.next_view_lib.append([next_pos, next_yaw_rad])
        # Log(f"Next View Library Generated. Total {len(self.next_view_lib)} viewpoints.", tag="ActiveManage")

    def nbv_select(self):
        if len(self.next_view_lib)<=0:
            print("No viewpoints in library. Back to the last viewpoint.")
            [next_pos, next_yaw] = self.previous_vps[-2]
            ### sometimes the previous viewpoint [-1] is a dead-end, get out the dead end by previous viewpoint [-2] ###
            return next_pos, next_yaw
        
        # clone the latest gaussians for rendering
        self.thread_lock_gs.acquire()
        self.gaussians = clone_obj(self.gaussians_frontend)
        self.gaussians.prune_only_object()
        self.thread_lock_gs.release()
        
        # compute GauSS-MI among all candidate viewpoints
        nbv_pos = None
        next_yaw = None
        best_gain = -1000.0
        for [pos_w, yaw] in self.next_view_lib:
            image_info, _ = self.compute_GauSS_MI(pos_w, yaw)

            pre_offset = 1.0 - self.inPreviousVps(pos_w, yaw)
            gain = image_info * pre_offset
            if gain > best_gain:
                best_gain = gain
                nbv_pos = pos_w
                next_yaw = yaw

        Log(f"NBV Selected. Gain {np.around(best_gain, decimals=2)} k, Pos {np.around(nbv_pos.transpose(), decimals=2)}, Yaw {np.around(next_yaw*180.0/np.pi, decimals=2)} deg", tag="ActiveManage")
        return nbv_pos, next_yaw

    def compute_GauSS_MI(self, pos, yaw):
        # world frame to robot frame
        ori_pose = self.getPoseMatrix(pos, yaw)
        pose = torch.from_numpy(np.linalg.inv(ori_pose)).to(device="cuda")
        self.viewpoint_cam.set_RT(pose[0:3, 0:3], pose[0:3, 3])

        # render the viewpoint
        render_pkg = render(self.viewpoint_cam, self.gaussians, self.pipeline_params, self.background)
        mutual_info = render_pkg["mutual_info"]
        mutual_info = mutual_info[0, :, :].detach().cpu().numpy()
        return int(np.sum(mutual_info)) /1000.0, render_pkg["render"]


    """
    # viewpoint tool functions
    """
    # append new viewpoint
    def append_viewpoint(self, pos_w, yaw_rad):
        yaw_rad = self.constrain_yaw_rad(yaw_rad)

        # 1. append `previous_vps_`        # in world frame, [pos, yaw(rad)]
        self.previous_vps.append([pos_w, yaw_rad])
        if len(self.previous_vps) > self.max_num_previous_vps:
            self.previous_vps.pop(0)

        # 2. publish desire pose to robot
        self.publish_set_pose(pos_w, yaw_rad)
        self.take_picture = self.take_picture+ 1
    
    def isValid(self, pos):
        # work space
        for i in range(3):
            if pos[i,0] < self.work_region[0,i] or pos[i,0] > self.work_region[1,i]:
                return False
        
        """ ### Robot Implementation ###
        # Collision Check for the new pose
        """
        return True

    def inPreviousVps(self, pos, yaw):
        if len(self.previous_vps) <= 5:
            return 0.0
        
        pre_discount = self.pre_discount_coeff
        for i in range(len(self.previous_vps)-2, 0, -1):
            [pos_i, yaw_i] = self.previous_vps[i]
            d_xy = np.linalg.norm(pos_i - pos) 
            d_yaw = abs(yaw_i - yaw)
            if d_xy < self.pre_max_pos_dist and d_yaw < self.pre_max_yaw_diff:
                return pre_discount
            pre_discount *= self.pre_discount_coeff

        return 0.0
    

    """
    # Robot Control Timer:
    # * communicate to agent
    # * send command pose and capture picture
    """
    def RobotTimer(self, event):
        if self.data_reader is None:
            Log('Data reader not initialized. Waiting for data reader ...', tag="ActiveManage")
            return
        if self.recon_done:
            if self.robot_running:
                Log('... Active Recon Done. Stop machine. ', tag="ActiveManage")
                self.robot_running = False
            self.take_picture = 0
            return
        if self.data_reader.dataDone:
            self.reconDone()
            Log('... No more data.', tag="ActiveManage")
            return
        if self.take_picture <= 0:
            Log('No more pictures to take. Waiting for NBV select ...', tag="ActiveManage")
            return
        
        """ ### Robot Implementation ###
        # 1. Robot motion planner
        # 2. Waiting for robot's motion to the set pose 
        """

        # pic taking
        self.take_an_image()
        self.take_picture = self.take_picture - 1
        self.frame_cnts += 1
        print()
        Log(f'No. {self.frame_cnts} Image Collected... ', tag="ActiveManage")
        return

    # take image and generate image-pose stack
    def take_an_image(self):
        received = False
        while not received:
            received, color, depth, ori_pose = self.data_reader[self.frame_cnts]
            
        # push to online 3dgs stack
        self.view_camera_stack.append(
            Camera.init_from_image_pose(
                self.data_reader,
                color, depth, ori_pose,
                self.frame_cnts
        ))


    """
    # recon Done Checker
    """
    """# call by backend, after reliability update"""
    def done_checker(self):
        if self.gaussians is None or not self.gaussians_inited or self.recon_done:
            return
        if self.frame_cnts < 2:
            return

        self.thread_lock_gs.acquire()
        num_gaussians = self.gaussians.get_reliability_1dim.shape[0]
        mask_reliable7 = (self.gaussians.get_reliability_1dim > self.done_prob).squeeze()
        self.thread_lock_gs.release()
        
        num_reliable7 = mask_reliable7.count_nonzero()
        done_percent_i = int((num_reliable7/num_gaussians)*10000)/100.0
        self.done_percent_smooth = self.done_percent_smooth*self.done_alpha + done_percent_i*(1-self.done_alpha)
        Log(f"Done Checker: Done Percnet: {self.done_percent_smooth}", tag="GauSS-MI")

        if self.done_percent_smooth > self.done_threshold:
            self.reconDone()
            Log("================== Active Recon Done! ==================", tag="GauSS-MI")
            return

    def reconDone(self):
        self.recon_done = True
        self.recon_done_pub.publish(Empty())


    """
    # tool functions
    """
    def publish_set_pose(self, pos, yaw):
        quat = Euler2Quaternion(np.array([[0.0], [0.0], [yaw]]))

        timestamp_now = rospy.Time.now()
        set_pose = PoseStamped()
        set_pose.header.frame_id = "camera"
        set_pose.header.stamp = timestamp_now
        set_pose.pose.position.x = pos[0, 0]
        set_pose.pose.position.y = pos[1, 0]
        set_pose.pose.position.z = pos[2, 0]
        set_pose.pose.orientation.x = quat.x
        set_pose.pose.orientation.y = quat.y
        set_pose.pose.orientation.z = quat.z
        set_pose.pose.orientation.w = quat.w
        self.set_pose_pub.publish(set_pose)

    def constrain_yaw_rad(self, yaw_rad):
        while yaw_rad > np.pi:
            yaw_rad -= 2*np.pi
        while yaw_rad < -np.pi:
            yaw_rad += 2*np.pi
        return yaw_rad
    
    def getPoseMatrix(self, pos_i, yaw):
        euler = np.array([[0.0], [0.0], [yaw]])

        T = np.zeros((4,4))
        T[0:3, 0:3] = rotConvert(Euler2Rot(euler))
        T[0:3, 3:4] = pos_i.reshape((3,1))
        T[3,3] = 1
        return T

