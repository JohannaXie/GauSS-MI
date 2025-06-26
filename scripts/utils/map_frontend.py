"""
# Frontend
# 1. Receive new `viewpoint`
# 2. Update mapping window
"""
import rospy
import time
import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getWorld2View2
from gui import gui_utils
from utils.save_utils import save_images
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.interest_region import InterestRegion

class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        # init and setup by main code
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None      # gui queue
        self.q_vis2main = None      # gui queue
        self.active_manager = None  # FSM manager
        self.gaussians = None

        # frames and key frames
        self.cameras = dict()               # store all frames
        self.current_window = []            # store id of windows, id => cur_frame_idx; recent large overlap keyframes
        self.occ_aware_visibility = {}      # visibility of current window  # update from backend

        # backend synchronize
        self.initialized = False
        self.reset = True
        self.pause = False                  # related to `self.q_vis2main`, pause and sync to backend
        self.requested_init = False         # backend state request
        self.requested_keyframe = 0

        # parameters
        self.device = "cuda:0"
        self.wait_image_rate = rospy.Rate(1000)
        self.keyframe_rate = rospy.Rate(5)
        self.pause_rate = rospy.Rate(100)

    def init_dataconfig(self, dataconfig):
        self.has_interest_region = dataconfig["Mapping"]["interested_region"]
        if self.has_interest_region:
            self.recon_region = InterestRegion(dataconfig)
        else:
            self.recon_region = None
            
        self.save_results = dataconfig["Save"]["save_results"]
        if self.save_results:
            self.save_dir = dataconfig["Save"]["save_dir"]

    def set_hyperparams(self):
        self.window_size = self.config["Training"]["window_size"]
        self.use_gui = self.config["Results"]["use_gui"]


    """
    # init the frame at the ground truth pose
    # request backend to init
    """
    def initialize(self, cur_frame_idx, viewpoint):
        Log(f'Initializing the frame No.{cur_frame_idx+1}...', tag="GauSS-MI",)
        self.initialized = True
        self.occ_aware_visibility = {}
        self.current_window = []
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # init backend
        self.request_init(cur_frame_idx, viewpoint)
        self.reset = False


    """
    # update window
    # remove frames which has little overlap with the current frame
    """
    def add_to_window(self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window):
        N_dont_touch = 2
        window = [cur_frame_idx] + window

        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(cur_frame_visibility_filter, occ_aware_visibility[kf_idx]).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]

        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.window_size:
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)

                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())

                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)
        return window

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def request_init(self, cur_frame_idx, viewpoint):
        msg = ["init", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        self.occ_aware_visibility = data[2]
        if (data[0]=="keyframe"):
            self.active_manager.gaussians_frontend = self.gaussians
            self.active_manager.gaussians_inited = True
            Log("Frontend: received gaussians from backend, pass to view manager", tag="OnlineGS",)##############
            self.active_manager.done_checker()
        

    """
    # Main RUN
    """
    def run(self):
        cur_frame_idx = 0
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while (not rospy.is_shutdown()):
            if self.use_gui:
                if self.q_vis2main.empty():
                    if self.pause:
                        Log(f'Paused by q_vis2main queue', tag="MonoGS",)
                        self.pause_rate.sleep()
                        continue
                else:
                    data_vis2main = self.q_vis2main.get()
                    self.pause = data_vis2main.flag_pause
                    if self.pause:
                        self.backend_queue.put(["pause"])
                        continue
                    else:
                        self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()

                """ 
                ### testing stop ###
                test_num = 10
                if cur_frame_idx > test_num:  
                    self.active_manager.reconDone()
                    Log(f"============ cur_frame_idx > {test_num} ! ============", tag="GauSS-MI")
                """

                if self.active_manager.recon_done:
                    Log(f'Frontend: recon done ', tag="OnlineGS",)
                    if self.save_results:
                        save_images(self.cameras, self.save_dir)
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue
                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                if self.requested_keyframe > 0:
                    self.keyframe_rate.sleep()
                    continue

                """
                # 1. receive image messages; preprocess
                """
                while (not rospy.is_shutdown()) and not self.active_manager.recon_done:
                    if len(self.active_manager.view_camera_stack) >0:
                        viewpoint = self.active_manager.view_camera_stack.pop(0)
                        break
                    self.wait_image_rate.sleep()

                if self.active_manager.recon_done:
                    Log(f'Frontend: recon done ', tag="OnlineGS",)
                    if self.save_results:
                        save_images(self.cameras, self.save_dir)
                    break

                if self.has_interest_region:
                    viewpoint.interest_region_mask = self.recon_region.get_interest_region_mask(viewpoint)
                self.cameras[cur_frame_idx] = viewpoint

                # initialization
                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue
                self.initialized = self.initialized or (len(self.current_window) == self.window_size)

                """
                # 2. update gui
                """
                if self.use_gui:
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            current_frame=viewpoint,
                            gtcolor=viewpoint.original_image,
                            gtdepth=viewpoint.depth,
                        )
                    )
                    current_window_dict = {}
                    current_window_dict[self.current_window[0]] = self.current_window[1:]
                    keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=clone_obj(self.gaussians),
                            current_frame=viewpoint,
                            keyframes=keyframes,
                            kf_window=current_window_dict,
                        )
                    )

                """
                # 3. update window
                """
                render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
                curr_visibility = (render_pkg["n_touched"] > 0).long()

                # update `self.current_window`, and remove low colap frames
                self.current_window = self.add_to_window(
                    cur_frame_idx, curr_visibility,
                    self.occ_aware_visibility, self.current_window,
                )
                # Log(f'Frontend: Frame No.{cur_frame_idx+1} Updated. ', tag="OnlineGS",)

                # request backend to update keyframe
                self.request_keyframe(cur_frame_idx, viewpoint, self.current_window)
                cur_frame_idx += 1

                toc.record()
                torch.cuda.synchronize()
                duration = tic.elapsed_time(toc)
                time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))      # throttle at 3fps when keyframe is added

            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)
                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1
                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False
                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break