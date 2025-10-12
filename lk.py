import torch
import theseus as th
from typing import Optional, List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt


class SparseLucasKanadeFlow(th.CostFunction):
    """
    Sparse Lucas-Kanade cost:
      - flow: theseus Vector containing flattened per-batch flow of shape (B, P*2)
      - I1, I2: images (B,C,H,W) where C is the number of channels (e.g., 1 for grayscale, 3 for RGB)
      - coords: keypoint pixel coords (B,P,2) in (x,y) pixel coordinates
      - window_size: odd integer (e.g., 3,5)
    Residuals: for each keypoint, for each window pixel, for each channel: I1_c(x) - I2_c(x + u)
    Jacobians: for each residual, J = -[I_x_c, I_y_c] (sampled at x+u)
    """
    def __init__(
        self,
        cost_weight: th.CostWeight,
        flow: th.Vector,
        I1: torch.Tensor,
        I2: torch.Tensor,
        coords: torch.Tensor,
        img_shape: Tuple[int,int],
        window_size: int = 3,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        self.flow = flow
        self.I1 = I1
        self.I2 = I2
        self.coords = coords
        self.img_shape = img_shape
        self.window_size = window_size
        self.P = coords.shape[1]
        self.C = I1.shape[1]

        # register optimization variable name that will be retrieved by Theseus internals
        self.register_optim_vars(["flow"])

    def _pixel_coords_to_grid(self, pts: torch.Tensor):
        """
        Convert pixel coords (x in [0, W-1], y in [0, H-1]) to grid_sample coords in [-1,1].
        pts: (...,2) with (x,y)
        """
        H, W = self.img_shape
        x = pts[..., 0]
        y = pts[..., 1]
        gx = (x / (W - 1)) * 2.0 - 1.0
        gy = (y / (H - 1)) * 2.0 - 1.0
        return torch.stack([gx, gy], dim=-1)

    def error(self) -> torch.Tensor:
        """
        Return residuals shaped (B, num_residuals) where num_residuals = P * W * W * C
        """
        B = self.I1.shape[0]
        P = self.P
        C = self.C
        W = self.window_size
        H, Wimg = self.img_shape

        # flow.tensor expected shaped (B, P*2)
        uv = self.flow.tensor.view(B, P, 2)

        # window offsets (dx, dy) in pixels
        half = (W - 1) // 2
        ys, xs = torch.meshgrid(
            torch.arange(-half, half+1, device=self.coords.device),
            torch.arange(-half, half+1, device=self.coords.device),
            indexing="ij"
        )
        offsets = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).float()  # (W*W,2) (dx,dy)

        # base pixel coords for all window pixels: (B, P, W*W, 2)
        base_coords = self.coords.unsqueeze(2) + offsets.view(1, 1, -1, 2)
        # apply flow -> warped sample locations in pixels
        warped_coords = base_coords + uv.unsqueeze(2)  # (B,P,W*W,2)

        # convert to normalized grid coords expected by grid_sample (order: x,y)
        base_grid = self._pixel_coords_to_grid(base_coords.view(B, -1, 2)).view(B, 1, -1, 2)
        warped_grid = self._pixel_coords_to_grid(warped_coords.view(B, -1, 2)).view(B, 1, -1, 2)

        # sample I1 and I2
        I1_vals = torch.nn.functional.grid_sample(
            self.I1, base_grid, align_corners=True, mode="bilinear", padding_mode="zeros"
        )  # (B,C,1,N) where N = P*W*W
        I2_vals = torch.nn.functional.grid_sample(
            self.I2, warped_grid, align_corners=True, mode="bilinear", padding_mode="zeros"
        )

        # reshape -> (B, P, W*W, C)
        Np = W * W
        I1_vals = I1_vals.view(B, C, P, Np).permute(0, 2, 3, 1)  # (B,P,Np,C)
        I2_vals = I2_vals.view(B, C, P, Np).permute(0, 2, 3, 1)

        residuals = (I1_vals - I2_vals)  # (B,P,Np,C)
        return residuals.reshape(B, P * Np * C)

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        B = self.I1.shape[0]
        P = self.P
        C = self.C
        W = self.window_size
        H, Wimg = self.img_shape
        Np = W * W  # residuals per point per channel
        N = P * Np * C  # total residuals

        uv = self.flow.tensor.view(B, P, 2)

        half = (W - 1) // 2
        ys, xs = torch.meshgrid(
            torch.arange(-half, half + 1, device=self.coords.device),
            torch.arange(-half, half + 1, device=self.coords.device),
            indexing="ij"
        )
        offsets = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).float()

        warped_coords = self.coords.unsqueeze(2) + uv.unsqueeze(2) + offsets.view(1, 1, -1, 2)
        warped_grid = self._pixel_coords_to_grid(warped_coords.view(B, -1, 2)).view(B, 1, -1, 2)

        # Sobel filters for each channel
        sobel_x_base = torch.tensor([[[-1., 0., 1.],
                                      [-2., 0., 2.],
                                      [-1., 0., 1.]]], device=self.I2.device, dtype=self.I2.dtype).unsqueeze(1)  # (1,1,3,3)
        sobel_y_base = torch.tensor([[[-1., -2., -1.],
                                      [0., 0., 0.],
                                      [1., 2., 1.]]], device=self.I2.device, dtype=self.I2.dtype).unsqueeze(1)  # (1,1,3,3)

        sobel_x = sobel_x_base.repeat(C, 1, 1, 1)  # (C,1,3,3)
        sobel_y = sobel_y_base.repeat(C, 1, 1, 1)  # (C,1,3,3)

        Ix = torch.nn.functional.conv2d(self.I2, sobel_x, padding=1, groups=C)
        Iy = torch.nn.functional.conv2d(self.I2, sobel_y, padding=1, groups=C)

        Ix_vals = torch.nn.functional.grid_sample(Ix, warped_grid, align_corners=True, mode="bilinear",
                                                  padding_mode="zeros")
        Iy_vals = torch.nn.functional.grid_sample(Iy, warped_grid, align_corners=True, mode="bilinear",
                                                  padding_mode="zeros")

        Ix_vals = Ix_vals.view(B, C, P, Np)  # (B, C, P, Np)
        Iy_vals = Iy_vals.view(B, C, P, Np)

        # Compute Jacobian per residual: (B, P, Np, C, 2)
        J_per_residual = torch.stack([-Ix_vals, -Iy_vals], dim=-1).permute(0, 2, 3, 1, 4)  # (B, P, Np, C, 2)

        # Create full Jacobian matrix (B, N, P*2)
        J_full = torch.zeros(B, N, P * 2, device=self.I2.device)

        # Fill in the blocks for each keypoint
        for i in range(P):
            start_idx = i * Np * C
            end_idx = start_idx + Np * C
            var_start = i * 2
            var_end = var_start + 2

            # Place the Jacobian for the current keypoint
            J_full[:, start_idx:end_idx, var_start:var_end] = J_per_residual[:, i, :, :, :].reshape(B, Np * C, 2)

        residuals = self.error()
        return [J_full], residuals

    def dim(self) -> int:
        return self.P * self.window_size * self.window_size * self.C

    def _copy_impl(self, new_name: Optional[str] = None) -> "SparseLucasKanadeFlow":
        return SparseLucasKanadeFlow(
            self.weight.copy(),
            self.flow.copy(),
            self.I1.clone(),
            self.I2.clone(),
            self.coords.clone(),
            self.img_shape,
            self.window_size,
            name=new_name,
        )


def demo_synthetic():
    torch.manual_seed(0)

    # 简单小示例
    B = 1
    H = 10
    W = 10
    P = 3  # 三个关键点
    C = 4  # 四层图像（例如RGBA）

    # 创建两个四通道图像
    I1 = torch.zeros(B, C, H, W, dtype=torch.float32)
    I2 = torch.zeros_like(I1)

    # 在I1的每个通道中放置三个亮点
    for c in range(C):
        I1[0, c, 2, 2] = 1.0
        I1[0, c, 5, 5] = 1.0
        I1[0, c, 7, 7] = 1.0

    # 在I2的每个通道中移动亮点
    for c in range(C):
        I2[0, c, 2, 3] = 1.0  # 向右1
        I2[0, c, 5, 6] = 1.0  # 向右1
        I2[0, c, 7, 8] = 1.0  # 向右1

    # 稀疏关键点坐标 (x, y)
    coords = torch.tensor([[[2.0, 2.0],
                            [5.0, 5.0],
                            [7.0, 7.0]]], dtype=torch.float32)  # (B, P, 2)

    # 初始光流猜测为零
    init_flow = torch.zeros(B, P, 2, dtype=torch.float32)
    flow_var = th.Vector(tensor=init_flow.view(B, -1), name="flow")

    # 权重
    weight = th.ScaleCostWeight(1.0)

    # 实例化成本函数
    lk_cost = SparseLucasKanadeFlow(
        weight,
        flow_var,
        I1,
        I2,
        coords,
        img_shape=(H, W),
        window_size=3,
        name="sparse_lk",
    )

    # 构建目标函数并添加成本
    obj = th.Objective()
    obj.add(lk_cost)

    lm = th.LevenbergMarquardt(obj, max_num_iterations=30)
    layer = th.TheseusLayer(lm)

    # 运行优化
    out = layer.forward(optimizer_kwargs={"verbose": True, "damping": 1e-3, "track_best_solution": False})

    # 检索优化后的变量
    optimized_flat = obj.get_optim_var("flow").tensor  # (B, P*2)
    optimized_flow = optimized_flat.view(B, P, 2)
    print("优化后的光流 (B,P,2):", optimized_flow)

    # 最终残差
    final_res = obj.error()
    print("最终残差形状:", final_res.shape)
    print("最终残差:", final_res)

    return optimized_flow


def demo_real_images(img1_path: str, img2_path: str, window_size: int = 21, max_corners: int = 100, quality_level: float = 0.01, min_distance: int = 10) -> torch.Tensor:
    # 读取RGB图像
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR_RGB)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR_RGB)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if img1 is None or img2 is None:
        raise ValueError(f"无法读取图像: {img1_path} 或 {img2_path}")

    # 调整图像大小以加快处理速度（可选）
    img1 = cv2.resize(img1, (640, 480))
    # img2 = cv2.resize(img2, (640, 480))
    # 左移 5 像素
    img2 = img1.copy()
    img2 = np.roll(img2, -5, axis=1)
    img2[:, -5:] = 0

    H, W = img1.shape[:2]
    C = 3  # RGB图像有3个通道

    # 转换为PyTorch张量，保持RGB通道，并添加批次维度
    # OpenCV使用BGR格式，转换为RGB并归一化到[0,1]
    I1_tensor = torch.from_numpy(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, C, H, W)
    I2_tensor = torch.from_numpy(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, C, H, W)

    I1_tensor = I1_tensor.to(device)
    I2_tensor = I2_tensor.to(device)

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
    coords = coords.unsqueeze(0).to(device)  # 添加批次维度 -> (1, P, 2)

    P = coords.shape[1]
    print(f"检测到 {P} 个关键点")

    # 初始光流猜测为零
    init_flow = torch.zeros(1, P, 2, dtype=torch.float32, device=device)
    flow_var = th.Vector(tensor=init_flow.view(1, -1), name="flow")

    # 权重
    weight = th.ScaleCostWeight(torch.tensor(1.0, device=device))

    # 实例化成本函数
    lk_cost = SparseLucasKanadeFlow(
        weight,
        flow_var,
        I1_tensor,
        I2_tensor,
        coords,
        img_shape=(H, W),
        window_size=window_size,
        name="real_image_lk",
    )

    # 构建目标函数
    obj = th.Objective()
    obj.add(lk_cost)

    # 创建Gauss-Newton优化器
    loss_list = []

    gn = th.GaussNewton(
        obj,
        max_iterations=100,
        step_size=1
    )

    layer = th.TheseusLayer(gn).to(device)

    # 运行优化
    print("开始优化...")
    solution, info  = layer.forward(optimizer_kwargs={"verbose": True, "damping": 1, "track_best_solution": True, "track_err_history": True})

    # 获取优化后的光流
    optimized_flat = obj.get_optim_var("flow").tensor
    optimized_flow = optimized_flat.view(1, P, 2)
    print("优化后的光流 (B, P, 2):", optimized_flow)

    # 可视化结果
    # visualize_results(img1, img2, coords[0].numpy(), optimized_flow[0].numpy())

    # 保存到 txt（每行一个数）
    np.savetxt("loss_log.txt", info.err_history.cpu().numpy(), fmt="%.6f")

    return optimized_flow



def visualize_results(img1, img2, keypoints, flow_vectors):
    plt.figure(figsize=(15, 10))

    # 绘制第一张图像和关键点
    plt.subplot(2, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='r', s=10)
    plt.title("image1 - keypoints")

    # 绘制第二张图像和关键点
    plt.subplot(2, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='r', s=10)
    plt.title("image2 - keypoints")

    # 绘制光流场
    plt.subplot(2, 2, 3)
    plt.imshow(img2, cmap='gray')
    for (x, y), (dx, dy) in zip(keypoints, flow_vectors):
        plt.arrow(x, y, dx, dy, color='cyan', width=0.5, head_width=3)
    plt.title("optical flow")

    # 绘制关键点移动
    plt.subplot(2, 2, 4)
    plt.imshow(img2, cmap='gray')
    moved_points = keypoints + flow_vectors
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='r', s=10, label='raw')
    plt.scatter(moved_points[:, 0], moved_points[:, 1], c='g', s=10, label='new')
    for i in range(len(keypoints)):
        plt.plot([keypoints[i, 0], moved_points[i, 0]],
                 [keypoints[i, 1], moved_points[i, 1]],
                 'y-', linewidth=0.8)
    plt.legend()
    plt.title("optical flow")

    plt.tight_layout()
    plt.savefig('optical_flow_results.png')
    plt.show()


# -------------------------------
# 主函数
# -------------------------------
if __name__ == "__main__":
    # print("运行合成示例...")
    # synthetic_flow = demo_synthetic()

    print("\n运行真实图像示例...")
    # 替换为您的图像路径
    img1_path = "img/1.JPG"
    img2_path = "img/2.JPG"

    real_flow = demo_real_images(img1_path, img2_path)
