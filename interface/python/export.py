import torch
import torch.nn as nn
import tensorrt as trt
import onnx
import os
from typing import Optional, Callable
from torchvision.models import resnet
import pnnx

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if gate is None: 
            self.gate = nn. ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = resnet.conv3x3(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = resnet.conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self. conv1(x)))
        x = self.gate(self.bn2(self. conv2(x)))
        return x


class ResBlock(nn. Module):
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
        out = self.gate(out)
        return out


class LETNet(nn.Module):
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
            self.conv_head = resnet.conv1x1(c2, 4)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2) / 255.
        # ================================== feature encoder
        block = self.block1(x)
        x1 = self.gate(self.conv1(block))
        # ================================== detector and descriptor head
        head = self.conv_head(x1)
        # score_map = torch.sigmoid(head[:, -1, :, :]).unsqueeze(1) * 10
        # descriptor = torch.sigmoid(head[:, 0:3, :, :])
        # descriptor = head

        descriptor = torch.sigmoid(head)
        descriptor = descriptor.permute(0, 2, 3, 1)
        # descriptor = descriptor[..., [2, 1, 0, 3]]
        descriptor = descriptor[..., [3, 2, 1, 0]]
        # descriptor = torch.sigmoid(descriptor)
        return descriptor*300.


class LETNet2(nn.Module):
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
            self.conv_head = resnet.conv1x1(c2, 4)

    def forward(self, x: torch.Tensor):
        # 修改：输入直接是标准格式 (B, C, H, W)，不需要permute
        x = x / 255.0
        block = self.block1(x)
        x1 = self.gate(self.conv1(block))
        head = self.conv_head(x1)
        descriptor = torch.sigmoid(head)
        # 输出也保持标准格式 (B, C, H, W)
        return descriptor * 300. 


def simplify_onnx(onnx_path):
    """使用onnx-simplifier简化模型"""
    try:
        import onnxsim
        print("Simplifying ONNX model...")
        model = onnx.load(onnx_path)
        model_simp, check = onnxsim. simplify(model)
        if check:
            onnx.save(model_simp, onnx_path)
            print("ONNX model simplified successfully")
        else:
            print("Warning: Simplified model validation failed, using original model")
    except ImportError:
        print("onnx-simplifier not installed, skipping simplification")
        print("Install with: pip install onnx-simplifier")


def export_onnx(model, H, W, save_path):
    device = torch.device("cpu")
    model.to(device).eval()

    input_names = ["image"]
    output_names = ["feature"]

    # 修改：输入格式改为标准的 (B, C, H, W)
    dummy_input = torch. randn(1, H, W, 3).to(device)

    print("Exporting to ONNX...")
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        # 先测试一下模型输出
        test_output = model(dummy_input)
        print(f"Output shape: {test_output.shape}")
        
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=11,
            export_params=True,
            do_constant_folding=True,
            dynamic_axes=None,
            verbose=False
        )
    print(f"ONNX model saved to {save_path}")
    
    # 验证ONNX模型
    try:
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed")
    except Exception as e: 
        print(f"Warning: ONNX model validation failed: {e}")
    
    # 简化ONNX模型
    simplify_onnx(save_path)


def export_tensorrt(onnx_path, trt_path, input_shape, fp16=True):
    """
    Export TensorRT engine from ONNX model
    
    Args:
        onnx_path: Path to ONNX model
        trt_path: Path to save TensorRT engine
        input_shape: Input shape (batch, C, H, W)  # 注意：现在是 (B, C, H, W) 格式
        fp16: Enable FP16 precision
    """
    logger = trt. Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # 创建网络定义
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # 配置解析器
    parser = trt.OnnxParser(network, logger)

    # 解析ONNX模型
    print(f"Parsing ONNX model from {onnx_path}...")
    with open(onnx_path, "rb") as model_file:
        model_data = model_file.read()
        if not parser.parse(model_data):
            print("Failed to parse ONNX model:")
            for error in range(parser.num_errors):
                print(f"  Error {error}: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX model")

    print("ONNX model parsed successfully")

    # 创建构建配置
    config = builder.create_builder_config()

    # 设置工作空间大小
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB

    # 启用FP16精度
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled")
    else:
        print("Using FP32 precision")

    # 验证输入形状
    print(f"Expected input shape: {input_shape}")
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"Network input {i}: name={input_tensor.name}, shape={input_tensor.shape}")

    # 构建引擎
    print("Building TensorRT engine...  (this may take a while)")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # 保存引擎
    with open(trt_path, "wb") as f:
        f.write(serialized_engine)
    print(f"TensorRT engine saved to {trt_path}")

    # 反序列化引擎
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    # 打印引擎信息
    print(f"Engine created with {engine.num_bindings} bindings")
    for i in range(engine.num_bindings):
        print(f"  Binding {i}: {engine.get_binding_name(i)}, shape={engine.get_binding_shape(i)}")
    
    return engine


def export_pipeline(model_path, H, W, batch_size=1, fp16=True):
    """
    Complete export pipeline
    
    Args: 
        model_path: Path to PyTorch model (. pth)
        H: Image height
        W: Image width
        batch_size: Batch size
        fp16: Enable FP16 precision for TensorRT
    """
    size_suffix = f"_{H}x{W}"
    
    onnx_path = f"./weights/letnet{size_suffix}.onnx"
    trt_path = f"./weights/letnet{size_suffix}.engine"
    ncnn_path = f"./weights/letnet{size_suffix}"
    # 修改：输入形状改为标准格式 (batch, C, H, W)
    input_shape = (batch_size, 3, H, W)

    # 加载模型
    print(f"Loading model from {model_path}...")
    net = LETNet(c1=8, c2=16, grayscale=False)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    print("Model loaded successfully")

    # 导出ONNX模型
    export_onnx(net, H, W, onnx_path)
    
    # 导出TensorRT引擎
    engine = export_tensorrt(onnx_path, trt_path, input_shape, fp16=fp16)
    
    # ncnn
    input_tensor = torch.rand(1, 3, H, W)
    net = LETNet2(c1=8, c2=16, grayscale=False)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    pnnx.export(net, ncnn_path, (input_tensor,))

    return engine


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export LETNet to ONNX and TensorRT')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to PyTorch model (.pth)')
    parser.add_argument('--height', type=int, default=480,
                        help='Input image height (default: 480)')
    parser.add_argument('--width', type=int, default=640,
                        help='Input image width (default: 640)')
    parser.add_argument('--batch', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--fp32', action='store_true',
                        help='Use FP32 instead of FP16 for TensorRT')
    parser.add_argument('--onnx-only', action='store_true',
                        help='Only export ONNX model, skip TensorRT')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os. makedirs("./weights", exist_ok=True)
    
    try:
        print("=" * 60)
        print("Starting model export pipeline...")
        print(f"  Model: {args.model}")
        print(f"  Input size: {args.height}x{args.width}")
        print(f"  Batch size: {args.batch}")
        print(f"  Precision: {'FP32' if args.fp32 else 'FP16'}")
        print("=" * 60)
        
        if args.onnx_only:
            # 只导出ONNX
            net = LETNet(c1=8, c2=16, grayscale=False)
            net.load_state_dict(torch.load(args.model, map_location='cpu'))
            net.eval()
            export_onnx(net, args.height, args.width, "./weights/letnet.onnx")
        else:
            # 完整导出
            engine = export_pipeline(
                args.model, 
                args.height, 
                args.width, 
                batch_size=args.batch,
                fp16=not args.fp32
            )
        
        print("=" * 60)
        print("Export completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print(f"Error occurred: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()