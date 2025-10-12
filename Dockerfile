# 使用 Ubuntu 20.04 作为基础镜像
FROM docker.1ms.run/ubuntu:20.04

# 设置时区和更新镜像源
RUN apt-get update && apt-get install -y tzdata \
    && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata

# 安装 OpenSSH, Git, Python 3.8, pip 和 OpenCV
RUN apt-get update && apt-get install -y \
    openssh-server \
    git \
    vim \
    python3.8 \
    python3.8-dev \
    python3-pip \
    libopencv-dev \
    python3-opencv \
    wget \
    curl \
    apt-transport-https \
    gnupg \
    && apt-get clean

# 升级 pip 
RUN python3.8 -m pip install --upgrade pip 


# 配置 SSH 服务
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
# RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config



# 开放 SSH 端口
EXPOSE 22

# 启动 SSH 服务
CMD /usr/sbin/sshd -D -e


# 创建目录 WORKDIR/code 并在目录下面git clone 代码
RUN mkdir -p /home/code
RUN cd /home/code && git clone https://github.com/linyicheng1/keypoint_bench.git

# 输出结果的目录
RUN mkdir -p /home/output


# 安装python 库
RUN pip3 install numpy~=1.24.4 torch~=2.1.2 torchvision~=0.16.2 \
    opencv-python~=4.9.0.80 thop~=0.1.1.post2209072238 kornia~=0.7.1 \
    h5py~=3.10.0 pillow~=10.2.0 tqdm~=4.66.1 pytorch_lightning openvino \
    tensorrt scikit-image pypose matplotlib
RUN pip install -U 'jsonargparse[signatures]>=4.27.7'
