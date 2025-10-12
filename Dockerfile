# 使用 Ubuntu 20.04 作为基础镜像
FROM docker.1ms.run/ubuntu:22.04

# 设置时区和更新镜像源
RUN apt-get update && apt-get install -y tzdata \
    && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata

# 安装核心系统依赖
RUN apt-get update && apt-get install -y \
    openssh-server \
    git \
    vim \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libopencv-dev \
    python3-opencv \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean

# 升级 pip 
RUN python3.10 -m pip install --upgrade pip 


# 配置 SSH 服务
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config

# 开放 SSH 端口
EXPOSE 22

# 创建目录 WORKDIR/code 
RUN mkdir -p /home/code

# 输出结果的目录
RUN mkdir -p /home/output

# 安装核心 Python 库 - 只包含实际使用的库
RUN pip3 install \
    numpy==2.2.6 \
    opencv-python-headless==4.12.0.88 \
    scipy

RUN pip3 install \
    torch==2.7.0 \
    torchvision==0.22.0 

RUN pip3 install \
    # theseus-ai==0.2.3 \
    albumentations \
    pypose==0.7.3 \
    kornia==0.8.1 \
    matplotlib==3.10.3 \
    pillow==11.2.1 

RUN pip3 install \
    scipy \
    scikit-image==0.25.2 \
    pandas==2.2.3 \
    open3d==0.19.0 \
    pyquaternion==0.9.9 \
    pybind11==3.0.1 \
    albucore==0.0.33 \
    kornia_rs==0.1.9 \
    torchkin==0.1.1 \
    torchlie==0.1.0

RUN apt install -y libsuitesparse-dev gfortran build-essential
RUN pip3 install theseus-ai

# 安装额外的必要库
RUN pip install -U 'jsonargparse[signatures]>=4.27.7'

# 解决可能的 OpenGL 问题
ENV PYOPENGL_PLATFORM egl

# 启动 SSH 服务
CMD /usr/sbin/sshd -D -e