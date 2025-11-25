# 使用 Ubuntu 18.04 ROS 
FROM docker.1ms.run/osrf/ros:melodic-desktop-full

# 设置时区和更新镜像源
RUN apt-get update && apt-get install -y tzdata \
    && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata

# 安装一些基础库
RUN apt-get update && apt-get install -y \
    git \
    vim \
    cmake \
    wget \
    zip \ 
    libgflags-dev \
    libgoogle-glog-dev \
    libssl-dev\
    build-essential \
    libgl1-mesa-dev \
    libglew-dev \
    libepoxy-dev \
    libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \ 
    libboost-all-dev \
    && apt-get clean

# CERES LIB 
RUN apt-get install -y libgoogle-glog-dev libgflags-dev libatlas-base-dev
RUN apt-get install -y libeigen3-dev libsuitesparse-dev

RUN mkdir /home/ceres
RUN cd /home/ceres && wget ceres-solver.org/ceres-solver-1.14.0.tar.gz

RUN cd /home/ceres && tar zxf ceres-solver-1.14.0.tar.gz && mkdir ceres-bin && cd ceres-bin && cmake ../ceres-solver-1.14.0 && make -j3 && make install 


# ============ 安装 CUDA 11.8 ============
# 安装 CUDA 11.8 依赖
RUN apt-get update && apt-get install -y \
    gnupg2 \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 添加 NVIDIA 仓库密钥
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb

# 更新并安装 CUDA 11.8
RUN apt-get update && apt-get install -y \
    cuda-toolkit-11-8 \
    cuda-libraries-11-8 \
    cuda-libraries-dev-11-8 \
    && rm -rf /var/lib/apt/lists/*

# 设置 CUDA 环境变量
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}


# ============ 手动安装 TensorRT ============
# 下载并安装 TensorRT
RUN mkdir -p /tmp/tensorrt && cd /tmp/tensorrt && \
    wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz && \
    tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz && \
    mv TensorRT-8.6.1.6 /opt/tensorrt && \
    rm -rf /tmp/tensorrt

# 设置 TensorRT 环境变量
ENV TENSORRT_PATH=/opt/tensorrt
ENV LD_LIBRARY_PATH=/opt/tensorrt/lib:${LD_LIBRARY_PATH}
ENV PATH=/opt/tensorrt/bin:${PATH}

# 创建符号链接以便系统找到 TensorRT
RUN ln -sf /opt/tensorrt/include /usr/include/tensorrt && \
    echo "/opt/tensorrt/lib" >> /etc/ld.so.conf.d/tensorrt.conf && \
    ldconfig

# 验证 TensorRT 安装
RUN cd /opt/tensorrt/samples/ && \
    make -j$(nproc) || echo "Samples build may fail but TensorRT is installed"

# 设置工作目录
WORKDIR /home/code

# 安装 cuDNN 8
RUN apt-get update && apt-get install -y \
    libcudnn8=8.9.2.*-1+cuda11.8 \
    libcudnn8-dev=8.9.2.*-1+cuda11.8

# 确保库路径
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 验证安装的脚本
RUN echo '#!/bin/bash' > /verify_install.sh && \
    echo 'echo "=== CUDA ==="' >> /verify_install.sh && \
    echo 'nvcc --version' >> /verify_install.sh && \
    echo 'echo "=== TensorRT ==="' >> /verify_install.sh && \
    echo 'ls /opt/tensorrt/include | head -5' >> /verify_install.sh && \
    echo 'find /opt/tensorrt -name "libnvinfer.so*" | head -3' >> /verify_install.sh && \
    chmod +x /verify_install.sh


# 模型导出脚本环境, 如果直接使用导出完成的模型就不需要了，自己导出则需要下面的部分

# ================= 安装 Python3.10 =================
RUN apt-get update && apt-get install -y \
    wget build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
    libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev \
    libbz2-dev liblzma-dev xz-utils \
    && cd /usr/src && \
    wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz && \
    tar xvf Python-3.10.13.tgz && \
    cd Python-3.10.13 && \
    ./configure --enable-optimizations && make -j$(nproc) && make altinstall && \
    cd /usr/src && rm -rf Python-3.10.13* 

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && rm get-pip.py && \
    /usr/local/bin/pip3.10 install --upgrade pip setuptools wheel


RUN /usr/local/bin/pip3.10 install sympy

# 设置 pip 国内源（清华）
RUN /usr/local/bin/pip3.10 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 PyTorch（CPU 版本）
RUN /usr/local/bin/pip3.10 install torch==1.11.0
RUN /usr/local/bin/pip3.10 install "numpy<2"

# 安装 Python TensorRT
RUN /usr/local/bin/pip3.10 install /opt/tensorrt/python/tensorrt-8.6.1-cp310-none-linux_x86_64.whl
RUN /usr/local/bin/pip3.10 install /opt/tensorrt/python/tensorrt_dispatch-8.6.1-cp310-none-linux_x86_64.whl
RUN /usr/local/bin/pip3.10 install /opt/tensorrt/python/tensorrt_lean-8.6.1-cp310-none-linux_x86_64.whl

RUN /usr/local/bin/pip3.10 install torchvision
RUN /usr/local/bin/pip3.10 install onnx


# EuRoC Dataset 
# docker run -it --gpus all -v /home/server/linyicheng/LETNET2/LET-NET2/VINS:/home/code/src -v /media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/euroc/:/home/data/ --net=host --env ROS_MASTER_URI=http://localhost:11311 --env ROS_IP=$(hostname -I | awk '{print $1}') let_vins:latest 
# rosrun 
