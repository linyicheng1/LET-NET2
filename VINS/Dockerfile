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

# docker run -it -v /home/server/linyicheng/LETNET2/VINS:/home/code/src let_vins:latest


