<div align="center">
    <h1>GauSS-MI: <br> 
    Gaussian Splatting Shannon Mutual Information for Active 3D Reconstruction</h1>
    <strong>RSS 2025</strong>
    <br>
    Yuhan Xie, Yixi Cai, Yinqiang Zhang, Lei Yang, and Jia Pan
    <br>
    <a href="https://www.roboticsproceedings.org/rss21/p030.pdf">Paper</a> 
    | <a href="https://arxiv.org/abs/2504.21067">arXiv</a> 
    | <a href="https://youtu.be/Qi7QpDyayKs?si=bhfjW1hmP8o2Trik">YouTube</a> 
    | <a href="https://www.bilibili.com/video/BV1rVNUziEFy">bilibili</a> 
</div>

### 📢 News
- **26 Jun. 2025**: GauSS-MI Code Release! 🔓
- **28 May. 2025**: Submodule [Differential Gaussian Rasterization with GauSS-MI](https://github.com/JohannaXie/diff-gaussian-rasterization-gaussmi) released. 

### 💡 Highlights
**GauSS-MI** is a real-time active view selection metric for 3D Gaussian Splatting, optimizing high-visual-quality 3D reconstruction.
<!-- <div align="center">
<img src="misc/GauSS-MI-overview-white.png" width=99% />
</div> -->

<!-- **Demo:** -->
<!-- <br> -->
<div align="center">
    <img src="misc/sim-recon.gif" width=49.7% />
    <img src="misc/sim-result.gif" width = 49.7% >
</div>
<div align="center">
    <img src="misc/rw-recon.gif" width=49.7% />
    <img src="misc/rw-result.gif" width = 49.7% >
</div>

### 🔗 BibTeX
If you find our code/work useful, please consider citing our work. Thank you!
```
@article{xie2025gaussmi,
  title     = {GauSS-MI: Gaussian Splatting Shannon Mutual Information for Active 3D Reconstruction},
  author    = {Yuhan Xie, Yixi Cai, Yinqiang Zhang, Lei Yang, and Jia Pan},
  journal   = {arXiv preprint arXiv:2503.02881},
  year      = {2025}
}
```

#### Acknowledgements
This project builds heavily on [MonoGS](https://github.com/muskie82/MonoGS) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). We thank the authors for their excellent work! If you use our code, please consider citing the papers as well.

## ⚙️ Environment Setup
#### 0. Requirements
* Most hardware and software requirements same as [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) Optimizer.
* Conda (recommended for easy setup)
* CUDA Toolkit 11.8
* [ROS1 Noetic](https://wiki.ros.org/noetic/Installation) + Ubuntu 20.04

#### 1. Clone the Repository
```bash
mkdir -p ~/ws_gaussmi/src && cd ~/ws_gaussmi/src    # ROS Workspace
git clone https://github.com/JohannaXie/GauSS-MI.git --recursive    # https
# git clone git@github.com:JohannaXie/GauSS-MI.git --recursive        # ssh
```

#### 2. Conda Setup
```bash
cd GauSS-MI
conda env create -f environment.yml
conda activate GauSS-MI
```

#### 3. Declare the Python path under your conda environment on the first line of `scripts/gs_map.py`, which could be `#!/opt/conda/envs/GauSS-MI/bin/python` or `#!/home/{YourUserName}/anaconda3/envs/GauSS-MI/bin/python`.

#### 4. ROS Setup
```bash
cd ~/ws_gaussmi
catkin build -DPYTHON_EXECUTABLE=/opt/conda/envs/GauSS-MI/bin/python    # Your python path under conda
```

## 🐳 Quick Setup with Docker

#### 1. Pull the docker image and build the container
```bash
# Pull Image
docker pull johanna17/gauss-mi:v1
# Run the Docker container
docker run -it -d --gpus all -e DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e QT_X11_NO_MITSHM=1 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    --shm-size=32G \
    --ipc=host \
    --network=host \
    --cap-add SYS_PTRACE \
    --cap-add SYS_ADMIN \
    --privileged \
    -v /dev:/dev \
    --name=GauSS-MI \
    -v $HOME/docker_envs/GauSS-MI:/home/do \
    -u do \
    -w /home/do \
    johanna17/gauss-mi:v1
# Go into the Container
docker exec -it GauSS-MI bash
# docker exec -itd GauSS-MI terminator  # Or use terminator
```

#### 2. [In Container] Activate ROS and Conda
```bash
source /opt/ros/noetic/setup.zsh
source /opt/conda/bin/activate 
/opt/conda/bin/conda init zsh
```

#### 3. [In Container] Clone the Repository and build
```bash
# Clone the repo
mkdir -p ~/ws_gaussmi/src && cd ~/ws_gaussmi/src    # ROS Workspace
git clone https://github.com/JohannaXie/GauSS-MI.git --recursive    # https
# Build the ros package
cd ~/ws_gaussmi
catkin config --extend /opt/ros/noetic
catkin build -DPYTHON_EXECUTABLE=/opt/conda/envs/GauSS-MI/bin/python
```

## 🚀 Quick Start
#### Use example rosbag
* Download the example robag [here](https://drive.google.com/drive/folders/1gnjs17tVXUiHU7rKAMtZDPmBt5acb1dn?usp=sharing).
* Launch GauSS-MI
    ```bash
    conda activate GauSS-MI
    cd ~/ws_gaussmi
    source devel/setup.zsh
    roslaunch gs_mapping gaussmi_rosbag_oildrum.launch
    ```
* Run rosbag in another terminal
    ```bash
    # source /opt/ros/noetic/setup.zsh
    rosbag play GauSS-MI_example1_oildrum.bag
    ```


## 🦾 Implement on Your Robot
#### ROS Publisher & Subscriber
* Publish your collected RGB image, depth image, and camera pose at initialization viewpoints and the next-best-view on rostopic, respectively:
    ```
    /camera/bgr
    /camera/depth
    /camera/pose
    ```
* Subscribe to the next-best-view pose on rostopic:
    ```
    /gaussmi/nbv_pose
    ```

#### Simulation or Real-World Robot Implementation
To implement our code on your robot with the motion planner and controller. You may need to customize the code in `scripts/active_recon/active_manage.py`. Specifically:
* Robot Initialization
* Collision Check
* Robot Motion Planner & Waiting for Robot's Navigation.

#### Customize Parameters
You may need to customize the parameters for your reconstruction environment in the configuration file `configs/eg_data_config/active.yaml`.

#### Launch GauSS-MI and Your Robot
* Launch GauSS-MI
    ```bash
    conda activate GauSS-MI
    cd ~/ws_gaussmi
    source devel/setup.zsh
    roslaunch gs_mapping gaussmi_active.launch
    ```
* Launch Your Robot

