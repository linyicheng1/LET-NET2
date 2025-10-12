import torch
import torch.nn as nn
import torch.optim as optim
from model import LETNet
from tartanair import TartanAir
from torch.utils.data import DataLoader
from typing import List, Tuple
import os
import theseus as th
from lk import SparseLucasKanadeFlow
from tqdm import tqdm
import cv2
import numpy as np
import torch.nn.functional as F

# def differentiable_sparse_lk(
#     desc1: torch.Tensor,     # (B, C, H, W)
#     desc2: torch.Tensor,     # (B, C, H, W)
#     coords: torch.Tensor,    # (B, P, 2)
#     mask: torch.Tensor,      # (B, P)
#     img_shape: Tuple[int, int],
#     gt_flow: torch.Tensor,   # (B, P, 2) 真实光流
#     noise_std: float = 3.0,  # 高斯噪声标准差
#     window_size: int = 21,
#     max_iter: int = 10,
#     damping: float = 1.0
# ) -> torch.Tensor:
#     B, P = mask.shape
#     device = desc1.device
#     optimized_flow = torch.zeros(B, P, 2, device=device)

#     for b in range(B):
#         valid = mask[b] > 0.5
#         if valid.sum() == 0:
#             continue

#         coords_b = coords[b][valid].unsqueeze(0)  # (1, Pv, 2)
#         Pv = coords_b.shape[1]

#         # 用 gt_flow + 噪声初始化
#         init_flow = gt_flow[b][valid] + noise_std * torch.randn_like(gt_flow[b][valid])
#         init_flow = init_flow.view(1, Pv * 2)

#         flow_var = th.Vector(tensor=init_flow, name="flow")
#         weight = th.ScaleCostWeight(torch.tensor(1.0, device=device))

#         lk_cost = SparseLucasKanadeFlow(
#             weight,
#             flow_var,
#             desc1[b:b+1],
#             desc2[b:b+1],
#             coords_b,
#             img_shape,
#             window_size,
#         )

#         obj = th.Objective()
#         obj.add(lk_cost)
#         gn = th.LevenbergMarquardt(obj, max_iterations=max_iter, step_size=1)
#         layer = th.TheseusLayer(gn).to(device)

#         out = layer.forward(optimizer_kwargs={
#             "verbose": False,
#             "damping": damping,
#             "track_best_solution": True
#         })

#         optimized_flat = obj.optim_vars["flow"].tensor
#         flow_b = optimized_flat.view(1, Pv, 2)
#         optimized_flow[b, valid] = flow_b[0]

#     return optimized_flow

def uncertainty_loss(pred_flow, gt_flow, uncertainty_map, coords, img_shape, mask=None):
    """
    计算稀疏光流的不确定度损失 (Laplace NLL)，并支持对 mask 加权平均。

    Args:
        pred_flow (torch.Tensor): 预测稀疏光流 (B, P, 2)
        gt_flow (torch.Tensor): 真实稀疏光流 (B, P, 2)
        uncertainty_map (torch.Tensor): 不确定度图 (B, 1, H, W) 或 (B, 2, H, W)
        coords (torch.Tensor): 关键点坐标 (B, P, 2)，像素单位 (x, y)
        img_shape (tuple): 图像尺寸 (H, W)
        mask (torch.Tensor or None): (B, P) 或 (B, P, 1) 的有效点掩码，1 表示有效

    Returns:
        torch.Tensor: 标量 NLL（masked average）
    """
    B, P, _ = pred_flow.shape
    H, W = img_shape
    global_mean = torch.mean(uncertainty_map)
    # 归一化 coords 到 [-1, 1]
    coords_norm = coords.clone()
    coords_norm[..., 0] = (coords[..., 0] / (W - 1)) * 2 - 1  # x
    coords_norm[..., 1] = (coords[..., 1] / (H - 1)) * 2 - 1  # y
    coords_norm = coords_norm.view(B, P, 1, 2)  # (B, P, 1, 2)

    # 从 uncertainty_map 采样 (B, C, P, 1)
    # grid_sample expects grid (B, H_out, W_out, 2) so here H_out=P, W_out=1
    sampled = F.grid_sample(uncertainty_map, coords_norm, mode='bilinear', align_corners=True)  # (B, C, P, 1)
    sampled = sampled.view(B, -1, P).permute(0, 2, 1)  # (B, P, C)
    scale = sampled + 1e-6  # 保证非零，前面已经用了 softplus，但再加 eps 更稳

    # 计算绝对误差
    error = torch.abs(pred_flow - gt_flow)  # (B, P, 2)

    # NLL 计算（Laplace）
    if scale.shape[-1] == 1:  # scalar uncertainty，对 u/v 共享 scale
        error_mean = error.mean(dim=-1, keepdim=True)  # (B, P, 1)
        nll_per_pt = error_mean / scale + torch.log(scale)  # (B, P, 1)
    else:  # per-component
        nll_comp = error / scale + torch.log(scale)  # (B, P, 2)
        nll_per_pt = nll_comp.mean(dim=-1, keepdim=True)  # (B, P, 1)

    # mask 支持
    if mask is not None:
        # mask -> (B, P, 1)
        if mask.ndim == 2:
            mask_ = mask.unsqueeze(-1).float()
        elif mask.ndim == 3:
            mask_ = mask.float()
        else:
            raise ValueError("mask must be (B,P) or (B,P,1)")
        nll_sum = (nll_per_pt * mask_).sum()
        denom = mask_.sum() + 1e-8
        return nll_sum / denom - global_mean * 0.1
    else:
        return nll_per_pt.mean() - global_mean * 0.1


def differentiable_sparse_lk(
    desc1: torch.Tensor,     # (B, C, H, W)
    desc2: torch.Tensor,     # (B, C, H, W)
    coords: torch.Tensor,    # (B, P, 2)
    mask: torch.Tensor,      # (B, P)
    img_shape: Tuple[int, int],
    gt_flow: torch.Tensor,   # (B, P, 2)
    noise_std: float = 3.0,  # 高斯噪声标准差
    window_size: int = 21,
    max_iter: int = 10,
    damping: float = 1.0,
    scale: float = 1.0       # 新增缩放参数
) -> torch.Tensor:
    B, P = mask.shape
    device = desc1.device
    optimized_flow = torch.zeros(B, P, 2, device=device)

    # 如果缩放因子不为 1，对特征图下采样
    if scale != 1.0:
        H, W = img_shape
        new_H, new_W = int(H * scale), int(W * scale)
        desc1_scaled = torch.nn.functional.interpolate(
            desc1, size=(new_H, new_W), mode="bilinear", align_corners=False
        )
        desc2_scaled = torch.nn.functional.interpolate(
            desc2, size=(new_H, new_W), mode="bilinear", align_corners=False
        )
    else:
        desc1_scaled, desc2_scaled = desc1, desc2

    for b in range(B):
        valid = mask[b] > 0.5
        if valid.sum() == 0:
            continue

        coords_b = coords[b][valid].unsqueeze(0)  # (1, Pv, 2)
        Pv = coords_b.shape[1]

        # 如果缩放，关键点坐标也需要缩放
        if scale != 1.0:
            coords_b = coords_b * scale
            img_shape_scaled = (int(img_shape[0] * scale), int(img_shape[1] * scale))
        else:
            img_shape_scaled = img_shape

        # 用 gt_flow + 噪声初始化
        init_flow = gt_flow[b][valid] + noise_std * torch.randn_like(gt_flow[b][valid])
        if scale != 1.0:
            init_flow = init_flow * scale  # 光流也缩放

        init_flow = init_flow.view(1, Pv * 2)

        flow_var = th.Vector(tensor=init_flow, name="flow")
        weight = th.ScaleCostWeight(torch.tensor(1.0, device=device))

        lk_cost = SparseLucasKanadeFlow(
            weight,
            flow_var,
            desc1_scaled[b:b+1],
            desc2_scaled[b:b+1],
            coords_b,
            img_shape_scaled,
            window_size,
        )

        obj = th.Objective()
        obj.add(lk_cost)
        gn = th.LevenbergMarquardt(obj, max_iterations=max_iter, step_size=1)
        # gn = th.GaussNewton(obj, max_iterations=max_iter, step_size=1, damping=1e-5)
        layer = th.TheseusLayer(gn).to(device)


        out = layer.forward(optimizer_kwargs={
            "verbose": False,
            "damping": damping,
            "track_best_solution": True
        })

        optimized_flat = obj.optim_vars["flow"].tensor
        flow_b = optimized_flat.view(1, Pv, 2)

        # 把光流缩放回原始大小
        if scale != 1.0:
            flow_b = flow_b / scale

        optimized_flow[b, valid] = flow_b[0]

        J_list, _ = lk_cost.jacobians()  # J_list[0]: (1, N, P*2)
        J_mat = J_list[0]  # (1, N, P*2)

    return optimized_flow, J_mat


def jacobian_orth_loss(J: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    B, N, D = J.shape
    H = torch.bmm(J.transpose(1, 2), J)  # (B, D, D)
    I = torch.eye(D, device=J.device).unsqueeze(0).expand(B, -1, -1)
    loss = ((H - I) ** 2).mean()
    return loss


def train_letnet_with_diff_lk(
        image_paths: List[str],
        num_epochs: int = 10,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_corners: int = 100,
        image_size: Tuple[int, int] = (640, 480),
        window_size: int = 21,
        num_lk_iter: int = 30,
        checkpoint_path: str = None,  # 可选：加载权重初始化,
        uncertainty_weight: float = 1e-6  # 不确定度损失的权重
):
    """
    Training LETNet with differentiable Lucas-Kanade flow estimation,
    using a tqdm progress bar.
    """
    # Initialize dataset
    dataset = TartanAir(
        path_lists=image_paths,
        max_corners=max_corners,
        quality_level=0.001,
        min_distance=10,
        image_size=image_size,
        gray=True
    )
    # dataset = ImageKeypointsDataset(
    #     image_paths=image_paths,
    #     max_corners=max_corners,
    #     quality_level=0.001,
    #     min_distance=10,
    #     image_size=image_size
    # )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, optimizer
    model = LETNet(c1=8, c2=16, grayscale=True).to(device)

    # 可选加载已有权重
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    sample_count = 0
    best_loss = float('inf')  # 用于保存最优模型

    for epoch in range(num_epochs):

        if epoch%10 == 0 and epoch > 0:
            dataset._build()
            print("Rebuilt dataset for new epoch")

        model.train()
        epoch_loss = 0.0
        num_valid_keypoints = 0

        # tqdm 进度条
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for i, (img_tensor, img_aug_tensor, keypoints_tensor, keypoints_aug_tensor, mask_tensor) in enumerate(pbar, start=1):
            # Move to device
            img_tensor = img_tensor.to(device)
            img_aug_tensor = img_aug_tensor.to(device)
            keypoints_tensor = keypoints_tensor.to(device)
            keypoints_aug_tensor = keypoints_aug_tensor.to(device)
            mask_tensor = mask_tensor.to(device)

            # Forward through LETNet
            score1, desc1 = model(img_tensor)
      
            _, desc2 = model(img_aug_tensor)

            # Ground truth flow
            gt_flow = keypoints_aug_tensor - keypoints_tensor

            bias_flow = torch.zeros_like(gt_flow)
            # if epoch % 4 == 0:
            #     bias_flow[:, :, 0] = bias_flow[:, :, 0] + 2
            # elif epoch % 4 == 1:
            #     bias_flow[:, :, 0] = bias_flow[:, :, 0] - 2
            # elif epoch % 4 == 2:
            #     bias_flow[:, :, 1] = bias_flow[:, :, 1] + 2
            # else:
            #     bias_flow[:, :, 1] = bias_flow[:, :, 1] - 2

            # Differentiable LK

            # pred_flow2, J2 = differentiable_sparse_lk(
            #     desc1, desc2, keypoints_tensor, mask_tensor, image_size,
            #     window_size=window_size,
            #     gt_flow=bias_flow,
            #     noise_std=0,
            #     scale=0.25,  # 使用缩放因子
            #     max_iter=num_lk_iter
            # )

            # pred_flow1, J1 = differentiable_sparse_lk(
            #     desc1, desc2, keypoints_tensor, mask_tensor, image_size,
            #     window_size=window_size,
            #     gt_flow=pred_flow2,
            #     noise_std=0,
            #     scale=0.5,  # 使用缩放因子
            #     max_iter=num_lk_iter
            # )

            pred_flow0, J0 = differentiable_sparse_lk(
                desc1, desc2, keypoints_tensor, mask_tensor, image_size,
                window_size=window_size,
                gt_flow=gt_flow,
                noise_std=2,
                max_iter=num_lk_iter
            )

            
            orth_loss = jacobian_orth_loss(J0)# + jacobian_orth_loss(J1) + jacobian_orth_loss(J2)
            orth_loss = orth_loss * 0.01  

            pred_flow = (pred_flow0) # + pred_flow1*0.5 + pred_flow2*0.25) / (1.0 + 0.5 + 0.25)

            # Masked loss
            mask_2d = mask_tensor.unsqueeze(-1).expand_as(gt_flow)
            flow_loss = (criterion(pred_flow, gt_flow) * mask_2d).sum() / (mask_2d.sum() + 1e-8)

            # 计算不确定度损失
            nll_loss = uncertainty_loss(
                pred_flow, gt_flow, score1, keypoints_tensor, image_size, mask_tensor
            )

            loss = flow_loss# + uncertainty_weight * nll_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update stats
            epoch_loss += loss.item() * img_tensor.size(0)
            num_valid_keypoints += mask_tensor.sum().item()
            sample_count += img_tensor.size(0)

            # Update tqdm postfix
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "flow_loss": f"{flow_loss.item():.4f}",
                "nll_loss": f"{nll_loss.item():.4f}",
                "valid_kps": int(mask_tensor.sum().item())
            })

            # Optional: save descriptor and augmented image for debugging
            if i % 50 == 0:  # only save first sample in batch
                desc_np = desc1.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                desc_np = (desc_np * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite("desc1.png", cv2.cvtColor(desc_np, cv2.COLOR_RGB2BGR))

                img_aug_np = img_aug_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                img_aug_np = (img_aug_np * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite("img_aug.png", cv2.cvtColor(img_aug_np, cv2.COLOR_RGB2BGR))

                score_np = (score1.squeeze(0).squeeze(0).detach().cpu().numpy() * 25).clip(0, 255).astype(np.uint8)
                cv2.imwrite("score1.png", score_np)

            # Visualize every 500 samples using the first item in the current batch
            # if sample_count % 500 == 0:
            #     # Use first sample in batch
            #     img_np = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            #     img_aug_np = (img_aug_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            #     kps_np = keypoints_tensor[0].cpu().numpy()
            #     kps_aug_np = keypoints_aug_tensor[0].cpu().numpy()
            #     gt_np = gt_flow[0].cpu().numpy()
            #     pred_np = pred_flow[0].cpu().detach().numpy()
            #     m = mask_tensor[0].cpu().numpy() > 0.5

            #     # Convert to BGR for cv2
            #     img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            #     aug_bgr = cv2.cvtColor(img_aug_np, cv2.COLOR_RGB2BGR)

            #     # GT matches visualization
            #     canvas_gt = np.hstack((img_bgr, aug_bgr))
            #     for i in range(len(m)):
            #         if m[i]:
            #             start = (int(kps_np[i, 0]), int(kps_np[i, 1]))
            #             gt_end = (int(kps_aug_np[i, 0]) + img_bgr.shape[1], int(kps_aug_np[i, 1]))
            #             cv2.circle(canvas_gt, start, 3, (0, 0, 255), -1)  # red
            #             cv2.circle(canvas_gt, gt_end, 3, (0, 0, 255), -1)
            #             cv2.line(canvas_gt, start, gt_end, (0, 255, 0), 1)  # green

            #     # Predicted matches visualization
            #     canvas_pred = np.hstack((img_bgr, aug_bgr))
            #     for i in range(len(m)):
            #         if m[i]:
            #             start = (int(kps_np[i, 0]), int(kps_np[i, 1]))
            #             pred_end = (int(kps_np[i, 0] + pred_np[i, 0]) + img_bgr.shape[1], int(kps_np[i, 1] + pred_np[i, 1]))
            #             cv2.circle(canvas_pred, start, 3, (0, 0, 255), -1)  # red
            #             cv2.circle(canvas_pred, pred_end, 3, (0, 0, 255), -1)
            #             cv2.line(canvas_pred, start, pred_end, (0, 255, 0), 1)  # green

            #     cv2.imwrite(f"gt_matches_epoch{epoch+1}_sample{sample_count}.png", canvas_gt)
            #     cv2.imwrite(f"pred_matches_epoch{epoch+1}_sample{sample_count}.png", canvas_pred)

        # Epoch summary
        avg_loss = epoch_loss / len(dataloader.dataset)
        avg_valid_kps = num_valid_keypoints / len(dataloader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg Loss: {avg_loss:.6f}, Avg Valid Keypoints: {avg_valid_kps:.2f}")

        # 保存 loss 最低的模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'letnet_best_model.pth')
            print(f"Saved new best model with loss {best_loss:.6f} -> letnet_best_model.pth")
        torch.save(model.state_dict(), 'last_model.pth')

    # Save trained model
    torch.save(model.state_dict(), 'letnet_model.pth')
    print("Model saved to letnet_model.pth")

    return model


def main():

    base_path = "/home/data/"
    path_list = [
        "abandonedfactory/Hard",
        # "abandonedfactory/Easy",
        # "abandonedfactory_night/Easy",
        "abandonedfactory_night/Hard",
        # "amusement/Easy",
        "amusement/Hard",
        # "carwelding/Easy",
        "carwelding/Hard",
        # "endofworld/Easy",
        "endofworld/Hard",
        # "gascola/Easy",
        "gascola/Hard",
        # "hospital/Easy",
        "hospital/Hard",
        # "japanesealley/Easy",
        "japanesealley/Hard",
        # "neighborhood/Easy",
        "neighborhood/Hard",
        # "ocean/Easy",
        "ocean/Hard",
        # "office/Easy",
        "office/Hard",
        # "office2/Easy",
        "office2/Hard",
        # "oldtown/Easy",
        "oldtown/Hard",
        # "seasidetown/Easy",
        "seasidetown/Hard",
        # "seasonsforest/Easy",
        "seasonsforest/Hard",
        # "seasonsforest_winter/Easy",
        "seasonsforest_winter/Hard",
        # "soulcity/Easy",
        "soulcity/Hard",
        # "westerndesert/Easy",
        "westerndesert/Hard"
    ]

    # path_list = ["abandonedfactory/Hard"]
    sequence_list = []
    for sequence in path_list:
        full_path = os.path.join(base_path, sequence)

        if os.path.exists(full_path) and os.path.isdir(full_path):
            # Get all immediate subdirectories
            subdirs = [d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))]

            for subdir in subdirs:
                subdir_path = os.path.join(full_path, subdir)
                sequence_list.append(subdir_path)
        else:
            print(f"Warning: Directory does not exist - {full_path}")


    checkpoint_path = None # "last_model.pth"

    # Filter valid image paths
    image_paths = [p for p in sequence_list if os.path.exists(p)]
    if not image_paths:
        print("No valid image paths provided.")
        return

    # Train the model
    model = train_letnet_with_diff_lk(
        image_paths=image_paths, 
        num_epochs=1000,
        batch_size=1,
        learning_rate=1e-3,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_corners=60,
        image_size=(320, 240),
        window_size=11,
        num_lk_iter=10,
        checkpoint_path=checkpoint_path
    )

if __name__ == "__main__":
    main()

