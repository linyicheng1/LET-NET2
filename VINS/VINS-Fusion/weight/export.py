from sympy.polys.groebnertools import lbp_key
import torch.nn.functional as F
import torch
import torch.nn as nn
import tensorrt as trt
import os
from typing import Optional, Callable
from torchvision.models import resnet

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

    # def forward(self, x: torch.Tensor):
    #     x = x.permute(0, 3, 1, 2) / 255.
    #     # ================================== feature encoder
    #     block = self.block1(x)
    #     x1 = self.gate(self.conv1(block))
    #     # ================================== detector and descriptor head
    #     head = self.conv_head(x1)
    #     # score_map = torch.sigmoid(head[:, -1, :, :]).unsqueeze(1) * 10
    #     # descriptor = torch.sigmoid(head[:, 0:3, :, :])

    #     descriptor = torch.sigmoid(head)
    #     descriptor = descriptor.permute(0, 2, 3, 1)
    #     descriptor = descriptor[..., [2, 1, 0, 3]]
    #     # descriptor = descriptor[..., [3, 2, 1, 0]]
    #     return descriptor*255.
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2) / 255.
        # to gray 
        x = x[:, 0:1]
        # ================================== feature encoder
        block = self.block1(x)
        x1 = self.gate(self.conv1(block))
        # ================================== detector and descriptor head
        head = self.conv_head(x1)
        # head[:, -1, :, :] = torch.sigmoid(head[:, -1, :, :]).unsqueeze(1)
        # head[:, 0:3, :, :] = torch.sigmoid(head[:, 0:3, :, :])

        descriptor = torch.sigmoid(head)

        descriptor = descriptor.permute(0, 2, 3, 1)
        # score_map = score_map.permute(0, 2, 3, 1)
        # descriptor = descriptor[..., [2, 1, 0, 3]]
        # descriptor = descriptor[..., [3, 2, 1, 0]]
        return descriptor*255.

# 编码器模型
net = LETNet(c1=8, c2=16, grayscale=True)
# net.load_state_dict(torch.load("../weights/letnet_model.pth"))
# net.load_state_dict(torch.load("../weights/letnet2.pth"))
net.load_state_dict(torch.load("last_model.pth"))
print("load success !")

def export_onnx(model, H, W, save_path):
    device = torch.device("cpu")
    model.to(device).eval()  # 确保模型在GPU上

    # 固定输入尺寸
    input_names = ["image"]
    output_names = ["feature"]
    # output_names = ["cov", "feature"]

    # 创建示例输入并确保在相同设备上
    dummy_input = torch.randn(1, H, W, 3).to(device)

    # 导出ONNX模型（固定尺寸）
    for name, module in model.named_modules():
        if list(module.parameters()):
            print(f"{name} on {next(module.parameters()).device}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=15,
            export_params=True,
            do_constant_folding=True,
            dynamic_axes=None  # 禁用动态尺寸
        )
    print(f"ONNX model saved to {save_path}")


def export_tensorrt(onnx_path, trt_path, input_shape, fp16=True):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)

    # 1. 创建网络定义
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # 2. 配置解析器
    parser = trt.OnnxParser(network, logger)

    # 3. 解析ONNX模型
    with open(onnx_path, "rb") as model_file:
        model_data = model_file.read()
        if not parser.parse(model_data):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX model")

    # 4. 创建构建配置
    config = builder.create_builder_config()

    # 启用FP16精度
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled")
    else:
        print("Using FP32 precision")

    # 5. 设置输入尺寸
    input_tensor = network.get_input(0)
    input_tensor.shape = input_shape

    # 6. 构建引擎
    # config.max_workspace_size = 2 << 30  # 2GB工作空间
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # 7. 保存引擎
    with open(trt_path, "wb") as f:
        f.write(serialized_engine)
    print(f"TensorRT engine saved to {trt_path}")

    # 反序列化引擎以便后续使用
    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(serialized_engine)


def export_pipeline(H, W, batch_size=1):
    onnx_path = "let2.onnx"
    trt_path = "let2_752_gray.engine"

    # 固定输入形状（批处理大小 x 通道数 x 高度 x 宽度）
    input_shape = (batch_size, H, W, 3)

    # 1. 导出ONNX模型（固定尺寸）
    export_onnx(net, H, W, onnx_path)
    # 2. 导出TensorRT引擎（固定尺寸）
    engine = export_tensorrt(onnx_path, trt_path, input_shape)
    return engine


if __name__ == "__main__":
    H, W = 480, 752
    # H, W = 608, 968
    # 确保输出目录存在
    # os.makedirs("../weights", exist_ok=True)
    try:
        print("Starting model export...")
        engine = export_pipeline(H, W, batch_size=1)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()

