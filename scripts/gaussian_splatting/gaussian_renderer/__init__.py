#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""
# Description: Update for 'Differential Gaussian Rasterization with GauSS-MI'
# Updated: Jun. 25, 2025
# Updated by: Yuhan (Johanna) Xie
# Email: yuhanxie@connect.hku.hk
"""

import math
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh

def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    loss_image=None,
    raster_reli=False,          # False: rasterize infogain;    "mutual_info" returns MI
                                # True:  rasterize reliability; "mutual_info" returns reli map for visualization
):
    """
    Render the scene.
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if pc.get_xyz.shape[0] == 0:
        return None

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity_in = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the Rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # check if the covariance is isotropic
        if pc.get_scaling.shape[-1] == 1:
            scales = pc.get_scaling.repeat(1, 3)
        else:
            scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by Rasterizer.
    shs = None
    colors_precomp = None
    if pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        shs = pc.get_features

    info_gain = pc.get_infogain
    if raster_reli:     ### only for visualization
        info_gain = pc.get_reliability

    # loss image
    if loss_image is None:
        loss_image = torch.zeros((viewpoint_camera.image_height, viewpoint_camera.image_width), dtype=torch.float32, device="cuda")

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth, opacity, n_touched, mutual_info, render_loss = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity_in,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        ### not unreliabilities here ### this variable is rasterized for mutual information
        unreliabilities=info_gain,
        loss_image=loss_image,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "n_touched": n_touched,
        "mutual_info": mutual_info,
        "render_loss": render_loss,
    }