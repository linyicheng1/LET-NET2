import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from torchvision import transforms as T
from typing import List, Tuple
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import bisect
import random
import pypose as pp
from bisect import bisect_right


class ImageKeypointsDataset(Dataset):
    """
    PyTorch Dataset class that:
    1. Reads image files from provided paths
    2. Converts images to PyTorch tensors
    3. Extracts up to 100 GFTT keypoints, padding to 100 if fewer
    4. Applies albumentations augmentations (geometric and photometric)
    5. Computes transformed keypoints, finds correspondences, and provides a mask
       for valid keypoints (strictly within image bounds: 0 <= x <= W-1, 0 <= y <= H-1)
    Returns: original image, augmented image, original keypoints, augmented keypoints, mask
    """
    def __init__(
        self,
        image_paths: List[str],
        max_corners: int = 80,
        quality_level: float = 0.001,
        min_distance: int = 10,
        image_size: Tuple[int, int] = (640, 480)
    ):
        self.image_paths = image_paths
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.image_size = image_size

        W, H = self.image_size

        # Define augmentation pipeline
        self.transform = A.Compose(
            [
                # Geometric augmentations
                A.Rotate(limit=3, p=0.0),
                A.RandomScale(scale_limit=0.01, p=0.0),
                A.ShiftScaleRotate(shift_limit=0.001, scale_limit=0.001, rotate_limit=1, p=1.0),
                # Photometric augmentations
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1.0),
                # A.GaussNoise(var_limit=(10.0, 50.0), p=0.0),
                # 保证增强后的图像尺寸与原始一致
                A.Resize(height=H, width=W, always_apply=True),
            ],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_and_preprocess_image(self, img_path: str) -> np.ndarray:
        # Read image in RGB format
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        return img

    def _extract_keypoints(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Convert to grayscale for GFTT
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(
            img_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance
        )
        if corners is None:
            keypoints = np.zeros((self.max_corners, 2), dtype=np.float32)
            mask = np.zeros(self.max_corners, dtype=np.float32)
        else:
            keypoints = corners.squeeze(1)  # Shape: (P, 2)
            P = len(keypoints)
            # Pad keypoints to max_corners
            if P < self.max_corners:
                padded = np.zeros((self.max_corners, 2), dtype=np.float32)
                padded[:P] = keypoints
                keypoints = padded
                mask = np.zeros(self.max_corners, dtype=np.float32)
                mask[:P] = 1.0
            else:
                keypoints = keypoints[:self.max_corners]
                mask = np.ones(self.max_corners, dtype=np.float32)
        return keypoints, mask

    def _augment_and_track_keypoints(
        self, img: np.ndarray, keypoints: np.ndarray, orig_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Apply augmentations
        augmented = self.transform(image=img, keypoints=keypoints)
        img_aug = augmented['image']
        keypoints_aug = np.array(augmented['keypoints'], dtype=np.float32)

        # Pad augmented keypoints to max_corners
        if len(keypoints_aug) < self.max_corners:
            padded = np.zeros((self.max_corners, 2), dtype=np.float32)
            padded[:len(keypoints_aug)] = keypoints_aug
            keypoints_aug = padded
        else:
            keypoints_aug = keypoints_aug[:self.max_corners]

        # Create mask for augmented keypoints with strict bounds check (0 <= x <= W-1, 0 <= y <= H-1)
        H, W = self.image_size
        aug_mask = np.zeros(self.max_corners, dtype=np.float32)
        valid_aug = (
            (keypoints_aug[:, 0] >= 0) & (keypoints_aug[:, 0] <= W - 1) &
            (keypoints_aug[:, 1] >= 0) & (keypoints_aug[:, 1] <= H - 1)
        )
        aug_mask[:len(valid_aug)] = valid_aug.astype(np.float32)

        # Combine masks: keypoint is valid only if it was originally valid and remains strictly in bounds
        final_mask = orig_mask * aug_mask

        # Optional: Clip augmented keypoints to bounds for visualization, but mask still filters them
        # keypoints_aug = np.clip(keypoints_aug, [0, 0], [W - 1, H - 1])

        return img_aug, keypoints_aug, final_mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Load and preprocess image
        img = self._load_and_preprocess_image(self.image_paths[idx])

        # Extract keypoints and initial mask
        keypoints, orig_mask = self._extract_keypoints(img)

        # Apply augmentations and track keypoints
        img_aug, keypoints_aug, final_mask = self._augment_and_track_keypoints(img, keypoints, orig_mask)

        # Convert to PyTorch tensors
        # Images: (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_aug_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float() / 255.0
        # Keypoints: (max_corners, 2)
        keypoints_tensor = torch.from_numpy(keypoints).float()
        keypoints_aug_tensor = torch.from_numpy(keypoints_aug).float()
        # Mask: (max_corners,)
        mask_tensor = torch.from_numpy(final_mask).float()

        return img_tensor, img_aug_tensor, keypoints_tensor, keypoints_aug_tensor, mask_tensor


def sort_files_by_prefix(folder: str):
    """辅助函数：按文件名前缀排序"""
    files = sorted(os.listdir(folder))
    return [os.path.join(folder, f) for f in files]


class TartanAir(Dataset):
    """
    TartanAir Dataset for sparse keypoints tracking.
    - First image: keypoints extracted using GFTT
    - Second image: light augmentation only
    - Keypoints in second image projected using pose & depth
    """

    def __init__(
        self,
        path_lists: List[str],
        max_corners: int = 80,
        quality_level: float = 0.001,
        min_distance: int = 10,
        image_size: Tuple[int, int] = (640, 480),
        frame_interval_range: Tuple[int, int] = (1, 1),
        gray: bool = False,
    ):
        self.pose_paths = []
        self.image_paths = []
        self.depth_paths = []
        self.image_pair_num = 0
        self.image_segment = []
        self.frame_interval_range = frame_interval_range
        self.gray = gray

        for path in path_lists:
            self.pose_paths.append(path + "/pose_left.txt")
            self.image_paths.append(sort_files_by_prefix(path + "/image_left/"))
            self.depth_paths.append(sort_files_by_prefix(path + "/depth_left/"))
            self.image_pair_num += len(self.image_paths[-1]) - 20
            self.image_segment.append(self.image_pair_num)

        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.image_size = image_size
        self.img_width, self.img_height = image_size

        self.poses = []
        for path in self.pose_paths:
            self.poses.append(self._load_tartanair_poses(path))
            
        K_default = np.array([
            [320.0, 0, 320.0],
            [0, 320.0, 240.0],
            [0, 0, 1]
        ], dtype=np.float32)

        self.K = self._scale_intrinsics(K_default, (640, 480), self.image_size)

        self.inv_K = np.linalg.inv(self.K)

        # Light augmentation only on second image
        if self.gray:
            self.light_transform = T.ColorJitter(brightness=0.3, contrast=0.3)
        else:
            self.light_transform = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)

        self._build()


    def _build(self, size = 1000):
        if self.image_pair_num < size:
            self.sample_pair_num = self.image_pair_num
            self.sample_ids = list(range(0, self.image_pair_num))
        else:
            self.sample_pair_num = size
            self.sample_ids = random.sample(range(self.image_pair_num), size)


    def _scale_intrinsics(
            self,
            K: np.ndarray,
            orig_size: Tuple[int, int],
            new_size: Tuple[int, int],
    ) -> np.ndarray:
        """根据图像缩放比例调整相机内参"""
        orig_w, orig_h = orig_size
        new_w, new_h = new_size
        sx, sy = new_w / orig_w, new_h / orig_h

        K_new = K.copy()
        K_new[0, 0] *= sx  # fx
        K_new[1, 1] *= sy  # fy
        K_new[0, 2] *= sx  # cx
        K_new[1, 2] *= sy  # cy
        return K_new

    def __len__(self):
        return self.sample_pair_num

    def _load_and_preprocess_image(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        if self.gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        return img

    def _extract_keypoints(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.gray:
            img_gray = img
        else:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        corners = cv2.goodFeaturesToTrack(
            img_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance
        )
        if corners is None:
            keypoints = np.zeros((self.max_corners, 2), dtype=np.float32)
            orig_mask = np.zeros(self.max_corners, dtype=np.float32)
        else:
            keypoints = corners.squeeze(1)
            P = len(keypoints)
            if P < self.max_corners:
                padded = np.zeros((self.max_corners, 2), dtype=np.float32)
                padded[:P] = keypoints
                keypoints = padded
            else:
                keypoints = keypoints[:self.max_corners]
            orig_mask = np.zeros(self.max_corners, dtype=np.float32)
            orig_mask[:P] = 1.0
        return keypoints, orig_mask

    def _load_depth(self, path: str) -> np.ndarray:
        return np.load(path)

    def _load_tartanair_poses(self, pose_file):
        """
        加载 TartanAir 位姿文件并进行坐标系转换
        :param pose_file: 位姿文件路径 (格式: tx ty tz qx qy qz qw)
        :return: pp.SE3 类型的位姿张量
        """
        # 读取原始位姿数据
        with open(pose_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        poses = []
        T = np.array([[0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [1, 0, 0, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        T_inv = np.linalg.inv(T)
        for line in lines:
            # 解析数据行
            data = list(map(float, line.split()))
            tx, ty, tz, qx, qy, qz, qw = data

            translation = np.array([tx, ty, tz], dtype=np.float32)
            q_original = torch.tensor([qx, qy, qz, qw])

            ################################################################
            # 构造 SE3 位姿
            ################################################################
            pose_tensor = torch.cat([
                torch.tensor(translation),
                q_original
            ])
            pose = T.dot(pp.SE3(pose_tensor).matrix().numpy()).dot(T_inv)

            poses.append(pp.mat2SE3(torch.from_numpy(pose)))

        return pp.SE3(torch.stack([p.tensor() for p in poses]))

    def _project_kps(self, pose0, pose1, kps0, margin=5, max_points=600):
        # 将关键点列表转换为张量 [N, 3]
        kps = torch.tensor(kps0, dtype=torch.float32)  # shape: (N, 3)
        mask = (
                (kps[:, 0] >= margin) &
                (kps[:, 0] < self.img_width - margin) &
                (kps[:, 1] >= margin) &
                (kps[:, 1] < self.img_height - margin) &
                (kps[:, 2] > 0)  # 深度必须为正
        )
        kps = kps[mask]

        # 坐标转换矩阵操作
        ones = torch.ones(kps.shape[0], 1)
        homo_coords = torch.cat([kps[:, :2], ones], dim=1)  # 齐次坐标

        # 反投影到图1相机坐标系 (深度用Z值)
        cam_coords = (torch.inverse(torch.from_numpy(self.K)) @ homo_coords.T).T * kps[:, 2].unsqueeze(1)
        # 坐标系转换到图2相机坐标系
        pose = pose0.Inv() * pose1
        cam2_coords = pose.Inv() * cam_coords
        # 投影到图2像素坐标系
        proj_coords = (torch.from_numpy(self.K) @ cam2_coords.T).T  # (N, 3)
        uv_coords = proj_coords[:, :2] / proj_coords[:, 2].clamp(min=1e-6).unsqueeze(1)

        # 有效性过滤（边界检查）
        valid_mask = (
                (uv_coords[:, 0] >= margin) &
                (uv_coords[:, 0] < self.img_width - margin) &
                (uv_coords[:, 1] >= margin) &
                (uv_coords[:, 1] < self.img_height - margin) &
                (proj_coords[:, 2] > 0)  # 深度必须为正
        )

        mask_kps0 = kps[valid_mask]
        kps_01 = uv_coords[valid_mask]

        mask = torch.zeros([max_points, 1]).to(kps_01.device)
        f_kps0 = torch.ones([max_points, 3]).to(kps_01.device)
        f_kps01 = torch.ones([max_points, 2]).to(kps_01.device)
        if kps_01.size(0) >= max_points:
            f_kps0 = mask_kps0[:max_points, :]
            f_kps01 = kps_01[:max_points, :]
            mask = torch.ones([max_points, 1]).to(kps_01.device)
        else:
            f_kps0[:kps_01.size(0), :] = mask_kps0
            f_kps01[:kps_01.size(0), :] = kps_01
            mask[:kps_01.size(0), :] = torch.ones([kps_01.size(0), 1]).to(kps_01.device)
        return f_kps0, f_kps01, mask
    
    
    def _project_keypoints(
        self,
        pose0: pp.SE3,
        pose1: pp.SE3,
        keypoints: np.ndarray,   # (N, 2) 像素坐标
        depth: np.ndarray,       # (H, W) 深度图
        orig_mask: np.ndarray    # (N,) 是否为有效点
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        向量化实现关键点从 cam0 投影到 cam1, 保持输入输出一一对应
        """
        device = torch.device("cpu")  # 或者 "cuda"
        K = torch.from_numpy(self.K).to(device)
        inv_K = torch.from_numpy(self.inv_K).to(device)

        W, H = self.image_size
        depth_H, depth_W = depth.shape
        N = keypoints.shape[0]

        # --- Step1: 从图像坐标映射到 depth 图索引 ---
        x = keypoints[:, 0] * (depth_W / W)
        y = keypoints[:, 1] * (depth_H / H)
        x_idx = np.clip(np.round(x).astype(np.int32), 0, depth_W - 1)
        y_idx = np.clip(np.round(y).astype(np.int32), 0, depth_H - 1)
        d = depth[y_idx, x_idx]  # (N,)

        # --- Step2: 构造齐次像素坐标并反投影 ---
        homo_coords = np.concatenate([keypoints, np.ones((N, 1), dtype=np.float32)], axis=1)  # (N, 3)
        homo_coords = torch.from_numpy(homo_coords).float().to(device)
        d_torch = torch.from_numpy(d).float().to(device)

        cam0_coords = (inv_K @ homo_coords.T).T * d_torch.unsqueeze(1)  # (N, 3)

        # --- Step3: 坐标系变换 cam0 -> cam1 ---
        relative_pose = pose0.Inv() * pose1
        cam0_hom = torch.cat([cam0_coords, torch.ones(N, 1, device=device)], dim=1)  # (N, 4)
        cam1_hom = (relative_pose.Inv().matrix().to(device) @ cam0_hom.T).T  # (N, 4)
        cam1_coords = cam1_hom[:, :3]  # (N, 3)

        # --- Step4: 投影到图像平面 ---
        proj_coords = (K @ cam1_coords.T).T  # (N, 3)
        uv_coords = proj_coords[:, :2] / proj_coords[:, 2].clamp(min=1e-6).unsqueeze(1)  # (N, 2)

        # --- Step5: 构造有效性掩码 ---
        valid_mask = (
            orig_mask.astype(bool) &              # 原始mask有效
            (d > 0) &                             # 深度为正
            (cam1_coords[:, 2].cpu().numpy() > 0) &  # 投影后仍在相机前方
            (uv_coords[:, 0].cpu().numpy() >= 0) &
            (uv_coords[:, 0].cpu().numpy() < W) &
            (uv_coords[:, 1].cpu().numpy() >= 0) &
            (uv_coords[:, 1].cpu().numpy() < H)
        ).astype(np.float32)  # (N,)

        # --- Step6: 组织输出 (保持和输入一一对应) ---
        keypoints_proj = np.zeros((N, 2), dtype=np.float32)
        keypoints_proj[valid_mask.astype(bool)] = uv_coords[valid_mask.astype(bool)].cpu().numpy()

        return keypoints_proj, valid_mask

    def __getitem__(self, index: int):

        index = self.sample_ids[index]
        
        interval = self.frame_interval_range[0] if self.frame_interval_range[0] == self.frame_interval_range[1] \
            else np.random.randint(*self.frame_interval_range)

        seq_id = bisect_right(self.image_segment, index)
        img_id = index - self.image_segment[seq_id - 1] if seq_id > 0 else index

        img_path1 = self.image_paths[seq_id][img_id]
        img_path2 = self.image_paths[seq_id][img_id + interval]
        depth_path1 = self.depth_paths[seq_id][img_id]

        img1 = self._load_and_preprocess_image(img_path1)
        img2 = self._load_and_preprocess_image(img_path2)
        depth1 = self._load_depth(depth_path1)

        keypoints, orig_mask = self._extract_keypoints(img1)
        pose0 = self.poses[seq_id][img_id]
        pose1 = self.poses[seq_id][img_id + interval]

        keypoints_aug, final_mask = self._project_keypoints(pose0, pose1, keypoints, depth1, orig_mask)

        # Convert images to tensors
        if self.gray:
            img1 = img1[:, :, None]
            img2 = img2[:, :, None]
        img_tensor = torch.from_numpy(img1).permute(2,0,1).float()/255.0
        img2_tensor = torch.from_numpy(img2).permute(2,0,1).float()/255.0

        if self.gray:
            img2_tensor = img2_tensor.repeat(3, 1, 1) 

        img_aug_tensor = self.light_transform(img2_tensor)

        if self.gray:
            img_aug_tensor = img_aug_tensor[0:1, :, :]

        keypoints_tensor = torch.from_numpy(keypoints).float()
        keypoints_aug_tensor = torch.from_numpy(keypoints_aug).float()
        mask_tensor = torch.from_numpy(final_mask).float()

        return img_tensor, img_aug_tensor, keypoints_tensor, keypoints_aug_tensor, mask_tensor
    


class TartanAirEVO(Dataset):
    """
    TartanAir Dataset for sparse keypoints tracking.
    - First image: keypoints extracted using GFTT
    - Second image: light augmentation only
    - Keypoints in second image projected using pose & depth
    """

    def __init__(
        self,
        path_lists: List[str],
        max_corners: int = 80,
        quality_level: float = 0.001,
        min_distance: int = 10,
        image_size: Tuple[int, int] = (640, 480),
        frame_interval_range: Tuple[int, int] = (1, 1),
    ):
        self.pose_paths = []
        self.image_paths = []
        self.depth_paths = []
        self.image_pair_num = 0
        self.image_segment = []
        self.frame_interval_range = frame_interval_range

        for path in path_lists:
            self.pose_paths.append(path + "/pose_left.txt")
            self.image_paths.append(sort_files_by_prefix(path + "/image_left/"))
            self.depth_paths.append(sort_files_by_prefix(path + "/depth_left/"))
            self.image_pair_num += len(self.image_paths[-1]) - 20
            self.image_segment.append(self.image_pair_num)

        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.image_size = image_size
        self.img_width, self.img_height = image_size

        self.poses = []
        for path in self.pose_paths:
            self.poses.append(self._load_tartanair_poses(path))
            
        K_default = np.array([
            [320.0, 0, 320.0],
            [0, 320.0, 240.0],
            [0, 0, 1]
        ], dtype=np.float32)

        self.K = self._scale_intrinsics(K_default, (640, 480), self.image_size)

        self.inv_K = np.linalg.inv(self.K)

        # Light augmentation only on second image
        self.light_transform = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)


    def _scale_intrinsics(
            self,
            K: np.ndarray,
            orig_size: Tuple[int, int],
            new_size: Tuple[int, int],
    ) -> np.ndarray:
        """根据图像缩放比例调整相机内参"""
        orig_w, orig_h = orig_size
        new_w, new_h = new_size
        sx, sy = new_w / orig_w, new_h / orig_h

        K_new = K.copy()
        K_new[0, 0] *= sx  # fx
        K_new[1, 1] *= sy  # fy
        K_new[0, 2] *= sx  # cx
        K_new[1, 2] *= sy  # cy
        return K_new

    def __len__(self):
        return self.image_pair_num

    def _load_and_preprocess_image(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        return img

    def _extract_keypoints(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(
            img_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance
        )
        if corners is None:
            keypoints = np.zeros((self.max_corners, 2), dtype=np.float32)
            orig_mask = np.zeros(self.max_corners, dtype=np.float32)
        else:
            keypoints = corners.squeeze(1)
            P = len(keypoints)
            if P < self.max_corners:
                padded = np.zeros((self.max_corners, 2), dtype=np.float32)
                padded[:P] = keypoints
                keypoints = padded
            else:
                keypoints = keypoints[:self.max_corners]
            orig_mask = np.zeros(self.max_corners, dtype=np.float32)
            orig_mask[:P] = 1.0
        return keypoints, orig_mask

    def _load_depth(self, path: str) -> np.ndarray:
        return np.load(path)

    def _load_tartanair_poses(self, pose_file):
        """
        加载 TartanAir 位姿文件并进行坐标系转换
        :param pose_file: 位姿文件路径 (格式: tx ty tz qx qy qz qw)
        :return: pp.SE3 类型的位姿张量
        """
        # 读取原始位姿数据
        with open(pose_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        poses = []
        T = np.array([[0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [1, 0, 0, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        T_inv = np.linalg.inv(T)
        for line in lines:
            # 解析数据行
            data = list(map(float, line.split()))
            tx, ty, tz, qx, qy, qz, qw = data

            translation = np.array([tx, ty, tz], dtype=np.float32)
            q_original = torch.tensor([qx, qy, qz, qw])

            ################################################################
            # 构造 SE3 位姿
            ################################################################
            pose_tensor = torch.cat([
                torch.tensor(translation),
                q_original
            ])
            pose = T.dot(pp.SE3(pose_tensor).matrix().numpy()).dot(T_inv)

            poses.append(pp.mat2SE3(torch.from_numpy(pose)))

        return pp.SE3(torch.stack([p.tensor() for p in poses]))

    def _project_kps(self, pose0, pose1, kps0, margin=5, max_points=600):
        # 将关键点列表转换为张量 [N, 3]
        kps = torch.tensor(kps0, dtype=torch.float32)  # shape: (N, 3)
        mask = (
                (kps[:, 0] >= margin) &
                (kps[:, 0] < self.img_width - margin) &
                (kps[:, 1] >= margin) &
                (kps[:, 1] < self.img_height - margin) &
                (kps[:, 2] > 0)  # 深度必须为正
        )
        kps = kps[mask]

        # 坐标转换矩阵操作
        ones = torch.ones(kps.shape[0], 1)
        homo_coords = torch.cat([kps[:, :2], ones], dim=1)  # 齐次坐标

        # 反投影到图1相机坐标系 (深度用Z值)
        cam_coords = (torch.inverse(torch.from_numpy(self.K)) @ homo_coords.T).T * kps[:, 2].unsqueeze(1)
        # 坐标系转换到图2相机坐标系
        pose = pose0.Inv() * pose1
        cam2_coords = pose.Inv() * cam_coords
        # 投影到图2像素坐标系
        proj_coords = (torch.from_numpy(self.K) @ cam2_coords.T).T  # (N, 3)
        uv_coords = proj_coords[:, :2] / proj_coords[:, 2].clamp(min=1e-6).unsqueeze(1)

        # 有效性过滤（边界检查）
        valid_mask = (
                (uv_coords[:, 0] >= margin) &
                (uv_coords[:, 0] < self.img_width - margin) &
                (uv_coords[:, 1] >= margin) &
                (uv_coords[:, 1] < self.img_height - margin) &
                (proj_coords[:, 2] > 0)  # 深度必须为正
        )

        mask_kps0 = kps[valid_mask]
        kps_01 = uv_coords[valid_mask]

        mask = torch.zeros([max_points, 1]).to(kps_01.device)
        f_kps0 = torch.ones([max_points, 3]).to(kps_01.device)
        f_kps01 = torch.ones([max_points, 2]).to(kps_01.device)
        if kps_01.size(0) >= max_points:
            f_kps0 = mask_kps0[:max_points, :]
            f_kps01 = kps_01[:max_points, :]
            mask = torch.ones([max_points, 1]).to(kps_01.device)
        else:
            f_kps0[:kps_01.size(0), :] = mask_kps0
            f_kps01[:kps_01.size(0), :] = kps_01
            mask[:kps_01.size(0), :] = torch.ones([kps_01.size(0), 1]).to(kps_01.device)
        return f_kps0, f_kps01, mask
    
    
    def _project_keypoints(
        self,
        pose0: pp.SE3,
        pose1: pp.SE3,
        keypoints: np.ndarray,   # (N, 2) 像素坐标
        depth: np.ndarray,       # (H, W) 深度图
        orig_mask: np.ndarray    # (N,) 是否为有效点
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        向量化实现关键点从 cam0 投影到 cam1, 保持输入输出一一对应
        """
        device = torch.device("cpu")  # 或者 "cuda"
        K = torch.from_numpy(self.K).to(device)
        inv_K = torch.from_numpy(self.inv_K).to(device)

        W, H = self.image_size
        depth_H, depth_W = depth.shape
        N = keypoints.shape[0]

        # --- Step1: 从图像坐标映射到 depth 图索引 ---
        x = keypoints[:, 0] * (depth_W / W)
        y = keypoints[:, 1] * (depth_H / H)
        x_idx = np.clip(np.round(x).astype(np.int32), 0, depth_W - 1)
        y_idx = np.clip(np.round(y).astype(np.int32), 0, depth_H - 1)
        d = depth[y_idx, x_idx]  # (N,)

        # --- Step2: 构造齐次像素坐标并反投影 ---
        homo_coords = np.concatenate([keypoints, np.ones((N, 1), dtype=np.float32)], axis=1)  # (N, 3)
        homo_coords = torch.from_numpy(homo_coords).float().to(device)
        d_torch = torch.from_numpy(d).float().to(device)

        cam0_coords = (inv_K @ homo_coords.T).T * d_torch.unsqueeze(1)  # (N, 3)

        # --- Step3: 坐标系变换 cam0 -> cam1 ---
        relative_pose = pose0.Inv() * pose1
        cam0_hom = torch.cat([cam0_coords, torch.ones(N, 1, device=device)], dim=1)  # (N, 4)
        cam1_hom = (relative_pose.Inv().matrix().to(device) @ cam0_hom.T).T  # (N, 4)
        cam1_coords = cam1_hom[:, :3]  # (N, 3)

        # --- Step4: 投影到图像平面 ---
        proj_coords = (K @ cam1_coords.T).T  # (N, 3)
        uv_coords = proj_coords[:, :2] / proj_coords[:, 2].clamp(min=1e-6).unsqueeze(1)  # (N, 2)

        # --- Step5: 构造有效性掩码 ---
        valid_mask = (
            orig_mask.astype(bool) &              # 原始mask有效
            (d > 0) &                             # 深度为正
            (cam1_coords[:, 2].cpu().numpy() > 0) &  # 投影后仍在相机前方
            (uv_coords[:, 0].cpu().numpy() >= 0) &
            (uv_coords[:, 0].cpu().numpy() < W) &
            (uv_coords[:, 1].cpu().numpy() >= 0) &
            (uv_coords[:, 1].cpu().numpy() < H)
        ).astype(np.float32)  # (N,)

        # --- Step6: 组织输出 (保持和输入一一对应) ---
        keypoints_proj = np.zeros((N, 2), dtype=np.float32)
        keypoints_proj[valid_mask.astype(bool)] = uv_coords[valid_mask.astype(bool)].cpu().numpy()

        return keypoints_proj, valid_mask

    def __getitem__(self, index: int):
        
        interval = self.frame_interval_range[0] if self.frame_interval_range[0] == self.frame_interval_range[1] \
            else np.random.randint(*self.frame_interval_range)

        seq_id = bisect_right(self.image_segment, index)
        img_id = index - self.image_segment[seq_id - 1] if seq_id > 0 else index

        img_path1 = self.image_paths[seq_id][img_id]
        img_path2 = self.image_paths[seq_id][img_id + interval]
        depth_path1 = self.depth_paths[seq_id][img_id]

        img1 = self._load_and_preprocess_image(img_path1)
        img2 = self._load_and_preprocess_image(img_path2)
        depth1 = self._load_depth(depth_path1)

        keypoints, orig_mask = self._extract_keypoints(img1)
        pose0 = self.poses[seq_id][img_id]
        pose1 = self.poses[seq_id][img_id + interval]

        keypoints_aug, final_mask = self._project_keypoints(pose0, pose1, keypoints, depth1, orig_mask)

        # Convert images to tensors
        img_tensor = torch.from_numpy(img1).permute(2,0,1).float()/255.0
        img2_tensor = torch.from_numpy(img2).permute(2,0,1).float()/255.0
        img_aug_tensor = self.light_transform(img2_tensor)

        keypoints_tensor = torch.from_numpy(keypoints).float()
        keypoints_aug_tensor = torch.from_numpy(keypoints_aug).float()
        mask_tensor = torch.from_numpy(final_mask).float()

        return img_tensor, img_aug_tensor, keypoints_tensor, keypoints_aug_tensor, mask_tensor
    

def visualize_keypoints(
        img: np.ndarray,
        keypoints: np.ndarray,
        img_aug: np.ndarray,
        keypoints_aug: np.ndarray,
        mask: np.ndarray,
        title: str = "Keypoints Visualization"
) -> None:
    """
    Visualize original and augmented images with valid keypoints based on mask.

    Args:
        img: Original image (H, W, C) in RGB
        keypoints: Original keypoints (100, 2)
        img_aug: Augmented image (H, W, C) in RGB
        keypoints_aug: Augmented keypoints (100, 2)
        mask: Validity mask (100,), 1.0 for valid, 0.0 for invalid
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Original image with valid keypoints
    ax1.imshow(img)
    valid = mask > 0.5  # Binary mask for valid keypoints
    if valid.sum() > 0:
        ax1.scatter(keypoints[valid, 0], keypoints[valid, 1], c='red', s=30, label='Valid Keypoints')
    ax1.set_title("Original Image")
    ax1.legend()
    ax1.axis('off')

    # Augmented image with valid keypoints
    ax2.imshow(img_aug)
    if valid.sum() > 0:
        ax2.scatter(keypoints_aug[valid, 0], keypoints_aug[valid, 1], c='blue', s=30, label='Valid Augmented Keypoints')
    ax2.set_title("Augmented Image")
    ax2.legend()
    ax2.axis('off')

    plt.suptitle(f"{title} ({valid.sum()} valid keypoints)")
    plt.tight_layout()
    plt.show()


def visualize_flow(
        img: np.ndarray,
        keypoints: np.ndarray,
        keypoints_aug: np.ndarray,
        mask: np.ndarray,
        title: str = "Keypoint Flow Visualization"
) -> None:
    """
    Visualize the flow (displacement) between valid original and augmented keypoints.

    Args:
        img: Original image (H, W, C) in RGB
        keypoints: Original keypoints (100, 2)
        keypoints_aug: Augmented keypoints (100, 2)
        mask: Validity mask (100,), 1.0 for valid, 0.0 for invalid
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(img)

    valid = mask > 0.5  # Binary mask for valid keypoints
    if valid.sum() > 0:
        for i in range(len(keypoints)):
            if valid[i]:
                x1, y1 = keypoints[i]
                x2, y2 = keypoints_aug[i]
                plt.arrow(x1, y1, x2 - x1, y2 - y1, color='yellow', head_width=5, head_length=5)
        plt.scatter(keypoints[valid, 0], keypoints[valid, 1], c='red', s=30, label='Original Keypoints')
        plt.scatter(keypoints_aug[valid, 0], keypoints_aug[valid, 1], c='blue', s=30, label='Augmented Keypoints')

    plt.title(f"{title} ({valid.sum()} valid keypoints)")
    plt.legend()
    plt.axis('off')
    plt.show()


def main():
    # Example image paths (replace with actual paths)
    image_paths = [
        "/home/linyi/VO/TheseusLK/img/1.JPG",
        "/home/linyi/VO/TheseusLK/img/2.JPG",
        # Add more image paths as needed
    ]

    # Filter valid image paths
    image_paths = [p for p in image_paths if os.path.exists(p)]
    if not image_paths:
        print("No valid image paths provided.")
        return

    # Initialize dataset
    dataset = ImageKeypointsDataset(
        image_paths=image_paths,
        max_corners=100,
        quality_level=0.01,
        min_distance=10,
        image_size=(640, 480)
    )

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Process one example
    for img_tensor, img_aug_tensor, keypoints_tensor, keypoints_aug_tensor, mask_tensor in dataloader:
        # Convert tensors to numpy for visualization
        img = img_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
        img_aug = img_aug_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
        print(img.shape, img_aug.shape)

        keypoints = keypoints_tensor.squeeze(0).numpy()  # (100, 2)
        keypoints_aug = keypoints_aug_tensor.squeeze(0).numpy()  # (100, 2)
        mask = mask_tensor.squeeze(0).numpy()  # (100,)

        print(f"Number of valid keypoints: {int(mask.sum())}")

        # Visualize keypoints
        visualize_keypoints(
            img.astype(np.uint8),
            keypoints,
            img_aug.astype(np.uint8),
            keypoints_aug,
            mask,
            title="Original vs Augmented Keypoints"
        )

        # Visualize flow
        visualize_flow(
            img.astype(np.uint8),
            keypoints,
            keypoints_aug,
            mask,
            title="Keypoint Flow"
        )
        break  # Only process one example for simplicity


def main2():
    path_lists = [
        "/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/tartanair/train/abandonedfactory/Hard/P000"
    ]

    # 过滤不存在的路径
    path_lists = [p for p in path_lists if os.path.exists(p)]
    if not path_lists:
        print("No valid dataset paths provided.")
        return

    # 初始化数据集
    dataset = TartanAir(
        path_lists=path_lists,
        max_corners=100,
        quality_level=0.01,
        min_distance=10,
        image_size=(640, 480),
        frame_interval_range=(1, 5)
    )

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 取一个样本测试
    for img_tensor, img_aug_tensor, keypoints_tensor, keypoints_aug_tensor, mask_tensor in dataloader:
        # 转 numpy
        img = img_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
        img_aug = img_aug_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
        keypoints = keypoints_tensor.squeeze(0).numpy()
        keypoints_aug = keypoints_aug_tensor.squeeze(0).numpy()
        mask = mask_tensor.squeeze(0).numpy()

        print(f"Image shape: {img.shape}, Aug image shape: {img_aug.shape}")
        print(f"Number of valid keypoints: {int(mask.sum())}")

        # 可视化关键点
        visualize_keypoints(
            img.astype(np.uint8),
            keypoints,
            img_aug.astype(np.uint8),
            keypoints_aug,
            mask,
            title="TartanAir Original vs Augmented Keypoints"
        )

        # 可视化光流
        visualize_flow(
            img.astype(np.uint8),
            keypoints,
            keypoints_aug,
            mask,
            title="TartanAir Keypoint Flow"
        )
        break  # 只展示一个 batch

if __name__ == '__main__':
    main2()
