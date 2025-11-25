import math
import cv2
import torch
from torch import nn
from torchvision.models import resnet
from typing import Optional, Callable
import cv2
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = resnet.conv3x3(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = resnet.conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x


class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            gate: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gate(out)

        return out


class LETNet(nn.Module):
    def __init__(self, c1: int = 8, c2: int = 16, grayscale: bool = False):
        super().__init__()
        self.gate = nn.ReLU(inplace=True)
        # ================================== feature encoder
        if grayscale:
            self.block1 = ConvBlock(1, c1, self.gate, nn.BatchNorm2d)
        else:
            self.block1 = ConvBlock(3, c1, self.gate, nn.BatchNorm2d)
        self.conv1 = resnet.conv1x1(c1, c2)
        # ================================== detector and descriptor head
        self.conv_head = resnet.conv1x1(c2, 3)

    def forward(self, x: torch.Tensor):
        # ================================== feature encoder
        block = self.block1(x)
        x1 = self.gate(self.conv1(block))
        # ================================== detector and descriptor head
        head = self.conv_head(x1)
        score_map = torch.sigmoid(head[:, -1, :, :]).unsqueeze(1) * 10
        descriptor = torch.sigmoid(head[:, 0:3, :, :])
        return score_map, descriptor


if __name__ == '__main__':
    img1_path = "img/1.JPG"
    img2_path = "img/2.JPG"

    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    # img1 = cv2.resize(img1, (640, 480))
    img1 = cv2.resize(img1, (640, 480))
    # img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR_RGB)
    img2 = img1.copy()
    img2 = np.roll(img2, -5, axis=1)
    img2[:, -5:] = 0

    if img1 is None or img2 is None:
        raise ValueError(f"无法读取图像: {img1_path} 或 {img2_path}")

    def preprocess(img):
        # resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 转换成 float32 并归一化
        img = img.astype("float32") / 255.0
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        # numpy -> torch
        img = torch.from_numpy(img).unsqueeze(0)  # [1,3,H,W]
        return img


    I1_tensor = torch.from_numpy(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, C, H, W)
    I2_tensor = torch.from_numpy(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, C, H, W)

    x1 = preprocess(img1)
    x2 = preprocess(img2)

    net = LETNet(c1=8, c2=16, grayscale=False)
    net.load_state_dict(torch.load("last_model.pth"))


    scores_map1, local_descriptor1 = net(x1)
    scores_map2, local_descriptor2 = net(x2)
    print(scores_map1.shape, local_descriptor1.shape)
    print(scores_map2.shape, local_descriptor2.shape)


    desc_np = local_descriptor1.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    desc_np = (desc_np * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite("desc.png", cv2.cvtColor(desc_np, cv2.COLOR_RGB2BGR))
    
    from lk import SparseLucasKanadeFlow
    import theseus as th

    H, W = img1.shape[:2]
    max_corners = 100
    quality_level = 0.01
    min_distance = 10
    window_size = 21


    # 使用GFTT检测关键点（在灰度图像上检测）
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(
        img1_gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance
    )

    if corners is None:
        raise RuntimeError("未检测到关键点，请调整参数")

    # 转换关键点为正确格式 (B, P, 2)
    coords = torch.from_numpy(corners).float().squeeze(1)  # (P, 2)
    coords = coords.unsqueeze(0)  # 添加批次维度 -> (1, P, 2)

    P = coords.shape[1]
    print(f"检测到 {P} 个关键点")

    # 初始光流猜测为零
    init_flow = torch.zeros(1, P, 2, dtype=torch.float32)
    flow_var = th.Vector(tensor=init_flow.view(1, -1), name="flow")
    # 权重
    weight = th.ScaleCostWeight(1.0)
    # 实例化成本函数
    lk_cost = SparseLucasKanadeFlow(
        weight,
        flow_var,
        local_descriptor1,
        local_descriptor2,
        coords,
        img_shape=(H, W),
        window_size=window_size,
        name="real_image_lk",
    )

    # 构建目标函数
    obj = th.Objective()
    obj.add(lk_cost)

    # 创建Gauss-Newton优化器
    gn = th.GaussNewton(
        obj,
        max_iterations=100,
        step_size=1,
    )

    layer = th.TheseusLayer(gn)

    # 运行优化
    print("开始优化...")
    out = layer.forward(optimizer_kwargs={"verbose": True, "damping": 1, "track_best_solution": True})

    # 获取优化后的光流
    optimized_flat = obj.get_optim_var("flow").tensor
    optimized_flow = optimized_flat.view(1, P, 2)
    print("优化后的光流 (B, P, 2):", optimized_flow)

