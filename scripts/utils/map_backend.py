"""
# Backend: Process new keyframe; Online Mapping
"""
import rospy
import time
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.map_utils import get_loss_mapping, get_loss_image

class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        # initialize and setup by main code
        self.config = config
        self.gaussians = None
        self.background = None
        self.cameras_extent = None
        self.pipeline_params = None
        self.opt_params = None
        self.frontend_queue = None
        self.backend_queue = None

        self.pause = False
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.iteration_count = 0
        self.last_sent = 0
        self.viewpoints = {}

        self.current_window = []
        self.occ_aware_visibility = {}

    def set_hyperparams(self):
        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (self.cameras_extent * self.config["Training"]["init_gaussian_extent"])

        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (self.cameras_extent * self.config["Training"]["gaussian_extent"])
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.isotropic_coeff = self.config["Training"]["isotropic_coeff"]

        self.touch_threshold = self.config["GauSS_MI"]["reli_touch_threshold"]
        self.min_loss = self.config["GauSS_MI"]["reli_min_loss"]
        self.lambda_L = self.config["GauSS_MI"]["reli_lambda_L"]
        self.lambda_T = self.config["GauSS_MI"]["reli_lambda_T"]
        self.loss_rgb_alpha = self.config["GauSS_MI"]["loss_rgb_alpha"]

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0):
        if frame_idx > 70:
            self.config["Dataset"]["pcd_downsample"] = 128
            return
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale
        )

    def reset(self):
        self.iteration_count = 0
        self.viewpoints = {}
        self.current_window = []
        self.occ_aware_visibility = {}
        self.gaussians.prune_points(self.gaussians.get_reliability >= 0)
        while not self.backend_queue.empty():
            self.backend_queue.get()


    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["n_touched"],
            )

            # loss
            loss_init = get_loss_mapping(self.config, image, depth, viewpoint)
            loss_init.backward()
            with torch.no_grad():       # Densification
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )
                if self.iteration_count == self.init_gaussian_reset or self.iteration_count == self.opt_params.densify_from_iter:
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    """
    # Run Twice: 1. iters; 2. last_iter = True
    """
    def map(self, current_window, last_iter=False, iters=1):
        if len(current_window) == 0:
            return

        # stack of all viewpoints in current_window
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        # stack of all viewpoints of "the rest of the keyframes" 
        random_viewpoint_stack = []
        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1
            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            # 1. iterate over the current window
            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(self.config, image, depth, viewpoint)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            # 2. random iterate over "the rest of the keyframes"
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(self.config, image, depth, viewpoint)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            # isotropic loss
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += self.isotropic_coeff * isotropic_loss.mean()
            loss_mapping.backward()

            # Deinsifying / Pruning Gaussians
            with torch.no_grad():
                if last_iter:
                    self.occ_aware_visibility = {}
                    for idx in range((len(current_window))):
                        kf_idx = current_window[idx]
                        n_touched = n_touched_acm[idx]
                        self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()
                    # # make sure we don't split the gaussians, break here.
                    return 
                
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (self.iteration_count % self.gaussian_update_every == self.gaussian_update_offset)
                if update_gaussian and self.iteration_count > 5:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                if (self.iteration_count % self.gaussian_reset) == 0 and (not update_gaussian):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
        return

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        if tag is None:
            tag = "sync_backend"
        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility]
        self.frontend_queue.put(msg)
        

    """
    ### Main RUN
    """
    def run(self):
        while (not rospy.is_shutdown()):
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, last_iter=True)
                    self.push_to_frontend()

            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "end":
                    self.push_to_frontend()
                    
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]

                    Log("Resetting the system")
                    self.reset()
                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(cur_frame_idx, viewpoint, init=True)
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe": 
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    self.current_window = data[3]
                    self.viewpoints[cur_frame_idx] = viewpoint

                    # 1. extend new gaussians based on new keyframe (MonoGS - Sec 3.3.2)
                    self.add_next_kf(cur_frame_idx, viewpoint)
                    # 2. mapping by window (MonoGS - Sec 3.3.3)
                    self.map(self.current_window, iters=self.mapping_itr_num)
                    self.map(self.current_window, last_iter=True)

                    self.update_reliability(viewpoint)
                    self.push_to_frontend("keyframe")

                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return

    def update_reliability(self, viewpoint):
        ## 1. compute loss on current keyframe
        render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
        loss_image = get_loss_image(render_pkg["render"], render_pkg["depth"], viewpoint, self.loss_rgb_alpha).detach() * self.lambda_L
        loss_image[loss_image < self.min_loss] = self.min_loss
        loss_image = -torch.log(loss_image)     # L' = -log(\lambda * L)

        ## 2. compute reliability for each touched gaussians (render loss with transparent T)
        render_out = render(viewpoint, self.gaussians, self.pipeline_params, self.background, loss_image=loss_image)
        logodds_zt = self.clip_logodds(render_out["render_loss"].detach() * self.lambda_T)
        logodds_zt.reshape(self.gaussians.get_xyz.shape[0], 4)
        n_touched = render_out["n_touched"].detach()
        logodds_zt[n_touched < self.touch_threshold] = 0.0

        ## 3. update unreliability   
        self.gaussians.input_logodds(logodds_zt)
    
    def clip_logodds(self, logodds):
        return torch.clamp(logodds, min=-100, max=100)