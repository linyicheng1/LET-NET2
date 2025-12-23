import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Optional, Callable, Tuple, List
from torchvision. models import resnet
import os
import argparse
from datetime import datetime

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer:  Optional[Callable[..., nn.Module]] = None):
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
        x = self.gate(self.bn1(self.conv1(x)))
        x = self.gate(self. bn2(self.conv2(x)))
        return x


class ResBlock(nn.Module):
    expansion:  int = 1

    def __init__(
            self,
            inplanes:  int,
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
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x:  torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None: 
            identity = self.downsample(x)
        out += identity
        out = self. gate(out)
        return out


class LETNet(nn. Module):
    def __init__(self, c1: int = 8, c2: int = 16, grayscale: bool = False):
        super().__init__()
        self.gate = nn.ReLU(inplace=True)
        if grayscale:
            self. block1 = ConvBlock(1, c1, self.gate, nn.BatchNorm2d)
        else:
            self. block1 = ConvBlock(3, c1, self.gate, nn.BatchNorm2d)
        self.conv1 = resnet.conv1x1(c1, c2)
        if grayscale: 
            self.conv_head = resnet.conv1x1(c2, 2)
        else:
            self.conv_head = resnet. conv1x1(c2, 4)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2) / 255.
        block = self.block1(x)
        x1 = self.gate(self.conv1(block))
        head = self.conv_head(x1)
        descriptor = torch.sigmoid(head)
        descriptor = descriptor.permute(0, 2, 3, 1)
        descriptor = descriptor[..., [3, 2, 1, 0]]
        return descriptor * 300.


class CornerTracking:
    def __init__(self, save_dir: str = "./tracking_results"):
        self.tracked_points: List[np.ndarray] = []
        self.tracked_points_history: List[List[np.ndarray]] = []
        self.prev_desc: Optional[np.ndarray] = None
        
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 创建descriptor子目录
        self.desc_dir = os.path.join(self.save_dir, "descriptors")
        os.makedirs(self.desc_dir, exist_ok=True)
        
        self.gftt_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=3,
            useHarrisDetector=False,
            k=0.04
        )
        
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
    
    def update(self, gray_img: np.ndarray, desc:  np.ndarray):
        if desc.dtype != np.uint8:
            desc = np.clip(desc, 0, 255).astype(np.uint8)
        
        if len(desc.shape) == 3 and desc.shape[2] > 3:
            desc_flow = desc[:, :, : 3]
        else:
            desc_flow = desc
            
        if len(self.tracked_points) == 0:
            self.tracked_points = self.extract_feature_gftt(gray_img)
            self.tracked_points_history = [[pt] for pt in self.tracked_points]
        else: 
            if len(self.tracked_points) > 0:
                tracked_points_array = np.array(self.tracked_points, dtype=np.float32).reshape(-1, 1, 2)
                
                tracked_points_new, status, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_desc,
                    desc_flow,
                    tracked_points_array,
                    None,
                    **self.lk_params
                )
                
                tracked = []
                tracked_history = []
                
                if status is not None:
                    for i, st in enumerate(status):
                        if st[0]: 
                            tracked.append(tracked_points_new[i][0])
                            self.tracked_points_history[i].append(tracked_points_new[i][0])
                            
                            if len(self.tracked_points_history[i]) > 5:
                                self.tracked_points_history[i].pop(0)
                            
                            tracked_history.append(self. tracked_points_history[i])
                
                add = self.extract_feature_gftt(gray_img, tracked)
                add_history = [[pt] for pt in add]
                
                self.tracked_points = tracked + add
                self.tracked_points_history = tracked_history + add_history
        
        self.prev_desc = desc_flow. copy()
    
    def extract_feature_gftt(
        self, 
        gray_img: np. ndarray, 
        existing_points: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        if gray_img is None or gray_img.size == 0:
            return []
        
        mask = np.ones_like(gray_img, dtype=np.uint8) * 255
        
        if existing_points is not None and len(existing_points) > 0:
            for pt in existing_points:
                pt_int = tuple(pt.astype(int))
                cv2.circle(mask, pt_int, 10, 0, -1)
        
        corners = cv2.goodFeaturesToTrack(
            gray_img,
            mask=mask,
            **self. gftt_params
        )
        
        if corners is None:
            return []
        
        detected_points = [corner. ravel() for corner in corners]
        
        return detected_points
    
    def save_descriptor(self, desc_map: np.ndarray, frame_id: int):
        """保存descriptor map图片"""
        if desc_map is not None:
            desc_vis = np.clip(desc_map[: , :, : 3], 0, 255).astype(np.uint8)
            desc_path = os.path.join(self.desc_dir, f"descriptor_{frame_id:06d}.jpg")
            cv2.imwrite(desc_path, desc_vis)
    
    def draw_tracking(self, img: np.ndarray) -> np.ndarray:
        """在图片上绘制跟踪点和轨迹"""
        vis_img = img.copy()
        
        # 绘制当前跟踪点（绿色）
        for pt in self.tracked_points:
            cv2.circle(vis_img, tuple(pt.astype(int)), 2, (0, 255, 0), -1)
        
        # 绘制轨迹（红色）
        for history in self.tracked_points_history:
            for i in range(1, len(history)):
                pt1 = tuple(history[i-1].astype(int))
                pt2 = tuple(history[i].astype(int))
                cv2.line(vis_img, pt1, pt2, (0, 0, 255), 1)
        
        return vis_img
    
    def get_tracked_points(self) -> List[np.ndarray]: 
        return self.tracked_points
    
    def get_tracked_history(self) -> List[List[np.ndarray]]:
        return self.tracked_points_history
    
    def reset(self):
        self.tracked_points = []
        self.tracked_points_history = []
        self.prev_desc = None


def main():
    parser = argparse.ArgumentParser(description='LETNet Corner Tracking')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to model weight file (. pth)')
    parser.add_argument('--input', '-i', type=str, default='0',
                        help='Input video path or camera index (default: 0 for webcam)')
    parser.add_argument('--output', '-o', type=str, default='./tracking_results',
                        help='Output directory (default: ./tracking_results)')
    parser.add_argument('--save-freq', type=int, default=1,
                        help='Save descriptor every N frames (default: 1)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display window (for headless mode)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (default:  auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model}")
    net = LETNet(c1=8, c2=16, grayscale=False)
    net.load_state_dict(torch.load(args. model, map_location=device))
    net.to(device)
    net.eval()
    print("Model loaded successfully!")
    
    # Initialize tracker
    tracker = CornerTracking(save_dir=args.output)
    
    # Open video or camera
    try:
        video_source = int(args.input)
    except ValueError:
        video_source = args.input
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video source:  {args.input}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height} @ {fps}fps")
    print(f"Saving results to: {args.output}")
    
    # Create video writer for tracking video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tracking_video_path = os.path.join(args.output, 'tracking_video.mp4')
    out_video = cv2.VideoWriter(
        tracking_video_path,
        fourcc, 
        fps if fps > 0 else 30, 
        (width, height)
    )
    
    print(f"Tracking video will be saved to: {tracking_video_path}")
    print(f"Descriptor images will be saved to: {os.path.join(args.output, 'descriptors/')}")
    
    frame_id = 0
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for GFTT extraction
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Prepare frame for network
            frame_tensor = torch. from_numpy(frame).unsqueeze(0).to(device)
            
            # Get descriptor map from network
            desc_map = net(frame_tensor)
            
            # Convert to numpy for OpenCV processing
            desc_map_np = desc_map[0].cpu().numpy()
            
            # Update tracker
            tracker.update(gray, desc_map_np)
            
            # Save descriptor map image (every N frames)
            if frame_id % args.save_freq == 0:
                tracker.save_descriptor(desc_map_np, frame_id)
            
            # Create visualization with tracking points and trajectories
            vis_img = tracker.draw_tracking(frame)
            
            # Write to tracking video
            out_video.write(vis_img)
            
            # Display (optional)
            if not args.no_display:
                cv2.imshow("Tracking Preview", vis_img)
                key = cv2.waitKey(1)
                if key == 27:  # ESC to exit
                    break
                elif key == ord('r'):  # 'r' to reset
                    tracker.reset()
            
            frame_id += 1
            
            if frame_id % 30 == 0:
                print(f"Processed {frame_id} frames, tracking {len(tracker.get_tracked_points())} points")
    
    # Cleanup
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Total frames processed: {frame_id}")
    print(f"Tracking video saved to: {tracking_video_path}")
    print(f"Descriptor images saved to: {os.path. join(args.output, 'descriptors/')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()