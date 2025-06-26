import os
from matplotlib import pyplot as plt
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log
from utils.rotation_utils import Rotation2Quaternion

def save_gaussians(gaussians, path, name):
    if name is None:
        return
    point_cloud_path = os.path.join(path, name)
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

def save_images(cameras, path, name="image"):
    Log("===== Saving images =====", tag="GauSS-MI")
    dir_jpg = os.path.join(path, "jpg_folder")
    mkdir_p(dir_jpg)

    dir_pose = os.path.join(path, "gt_pose.txt")
    file_pose = open(dir_pose, "w")

    for idx in range(len(cameras)):
        if idx == 0:
            continue
        image_name = str(idx).zfill(3) + ".jpg"

        color = cameras[idx].original_image.cpu().numpy().transpose((1, 2, 0))
        plt.imsave(os.path.join(dir_jpg, image_name), color)
        
        pos = cameras[idx].T_gt_ori
        Rot = cameras[idx].R_gt_ori
        quat = Rotation2Quaternion(Rot)
        file_pose.write(str(pos[0])+" "+str(pos[1])+" "+str(pos[2])+" "+str(quat.x)+" "+str(quat.y)+" "+str(quat.z)+" "+str(quat.w)+"\n")
