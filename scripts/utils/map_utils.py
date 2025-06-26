import torch


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda")
    conv_x = torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda")
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c)
    img_grad_h = normalizer * torch.nn.functional.conv2d(p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c)
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c)
    img_grad_h = torch.nn.functional.conv2d(p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c)

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def get_loss_mapping(config, image, depth, viewpoint):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()
    gt_depth = torch.from_numpy(viewpoint.depth).to(dtype=torch.float32, device=image.device)[None]

    interest_region_mask = viewpoint.interest_region_mask
    # image
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask *interest_region_mask - gt_image * rgb_pixel_mask *interest_region_mask)
    # depth
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    l1_depth = torch.abs(depth * depth_pixel_mask *interest_region_mask - gt_depth * depth_pixel_mask *interest_region_mask)

    # only l1 loss
    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


def get_loss_image(image, depth, viewpoint, loss_rgb_alpha=1.0):
    gt_image = viewpoint.original_image.cuda()
    gt_depth = torch.from_numpy(viewpoint.depth).to(dtype=torch.float32, device=image.device)[None]

    interest_region_mask = viewpoint.interest_region_mask
    rgb_boundary_threshold = 0.01
    # image
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask *interest_region_mask - gt_image * rgb_pixel_mask *interest_region_mask)
    l1_rgb = l1_rgb.mean(dim=0, keepdim=True)
    # depth
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    l1_depth = torch.abs(depth * depth_pixel_mask *interest_region_mask - gt_depth * depth_pixel_mask *interest_region_mask)

    loss_image = loss_rgb_alpha * l1_rgb + (1 - loss_rgb_alpha) * l1_depth
    return loss_image


