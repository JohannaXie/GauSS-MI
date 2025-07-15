#!/opt/conda/envs/GauSS-MI/bin/python

import rospy
from active_recon.active_manage import ActiveReconFSM
from active_recon.data_reader import load_data

import os
import time
from datetime import datetime
import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.save_utils import save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.map_backend import BackEnd
from utils.map_frontend import FrontEnd

class SLAM:
    def __init__(self, config, dataconfig):

        """
        # LOAD CONFIGURATION
        """
        # load configuration
        self.use_gui = config["Results"]["use_gui"]
        self.save_results = dataconfig["Save"]["save_results"]

        # gaussian splatting parameters
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        sh_degree = 3 if config["Training"]["spherical_harmonics"] else 0
        bg_color = [1, 1, 1] if config["model_params"]["white_background"] else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        """
        # Initializations
        """
        # initialize GaussianModel (maximum sh degree, config)
        self.gaussians = GaussianModel(sh_degree=sh_degree, config=config, dataconfig=dataconfig)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(opt_params)

        # initialize Active Manager and Data Reader
        self.active_manager = ActiveReconFSM(config=config, dataconfig=dataconfig)
        self.data_reader = load_data(model_params, config=dataconfig)
        self.active_manager.init_data_reader(self.data_reader)

        # initialize multiprocessing queues
        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()
        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        # initialize frontend and backend
        self.frontend = FrontEnd(config)
        self.backend = BackEnd(config)

        self.frontend.background = self.background
        self.frontend.pipeline_params = pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()
        self.frontend.init_dataconfig(dataconfig)
        self.frontend.active_manager = self.active_manager

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = pipeline_params
        self.backend.opt_params = opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.set_hyperparams()

        if self.use_gui:
            self.params_gui = gui_utils.ParamsGUI(
                pipe=pipeline_params,
                background=self.background,
                gaussians=self.gaussians,
                q_main2vis=q_main2vis,
                q_vis2main=q_vis2main,
            )

        """
        # Multiprocess START
        """
        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui, dataconfig["Save"]["save_dir"]))
            gui_process.start()
            time.sleep(5)

        backend_process.start()
        self.frontend.run()
        backend_queue.put(["pause"])

        # SLAM ends
        end = torch.cuda.Event(enable_timing=True)
        end.record()                        
        torch.cuda.synchronize()

        """
        # Multiprocess END
        """
        if self.save_results:
            save_dir = dataconfig["Save"]["save_dir"]
            self.gaussians = self.frontend.gaussians
            while (not frontend_queue.empty()) and (not rospy.is_shutdown()):
                frontend_queue.get()
            # sync gaussians with backend
            backend_queue.put(["end"])
            while (not rospy.is_shutdown()):
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()
                if data[0] == "sync_backend" and frontend_queue.empty():
                    self.gaussians = data[1]
                    break
            save_gaussians(self.gaussians, save_dir, "final")
        
        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread", tag="OnlineGS")

        if self.use_gui:
            wait_sec = int(config["Results"]["wait_sec"])
            for i in range(wait_sec):
                if rospy.is_shutdown():
                    break
                if i % 30 == 0:
                    Log(f'Sleeping for checking... {int(wait_sec - i)} seconds left ...', tag="GauSS-MI")
                time.sleep(1)
        
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread", tag="OnlineGS")
        
        rospy.on_shutdown(rospy_shutdown_callback)
        if rospy.is_shutdown():
            if gui_process.is_alive:
                gui_process.terminate()
                gui_process.join()
            if backend_process.is_alive:
                backend_process.terminate()
                backend_process.join()

    def run(self):
        pass

def rospy_shutdown_callback():
    Log("ROS Node Shutdown", tag="OnlineGS")

if __name__ == "__main__":
    mp.set_start_method("spawn")

    config_path = rospy.get_param('/gaussian_node/config_path')
    dataconfig_path = rospy.get_param('/gaussian_node/dataconfig_path')
    package_path = rospy.get_param('/gaussian_node/package_path')

    Log(f'Load Configuration from {config_path}', tag="GauSS-MI")
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    with open(dataconfig_path, "r") as yml:
        dataconfig = yaml.safe_load(yml)

    dataconfig["Save"]["save_dir"] = None
    if dataconfig["Save"]["save_results"]:
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = os.path.join(package_path, "results", dataconfig["Save"]["save_folder"], current_datetime) 
        mkdir_p(save_dir)
        dataconfig["Save"]["save_dir"] = save_dir
        Log("saving results in " + save_dir)

    rospy.init_node('ActiveRecon_node', anonymous=True)     # initialize ROS
    Log("ROS Node Inited", tag="OnlineGS")

    slam = SLAM(config, dataconfig)
    slam.run()
    Log("Done.", tag="OnlineGS")
    rospy.spin()
