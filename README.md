# LET-NET2：An end-to-end lightweight CNN designed for sparse corner extraction and tracking.

LET-NET2 transforms the traditional LK optical flow process into a neural network layer by computing its derivatives, enabling end-to-end training of sparse optical flow. It preserves the original lightweight network structure and, during training on simulated datasets, autonomously learns capabilities such as edge orientation extraction and active enhancement of weak-texture regions. As a result, it demonstrates stronger tracking performance under challenging conditions, including dynamic lighting, weak textures, low illumination, and underwater blur.

## Tracking Region Comparison for Improved Performance in Weak-Texture and Low-Light Conditions


<table align="center">
  <tr>
    <th style="text-align:center;font-size:16px;">Original LK Optical Flow</th>
    <th style="text-align:center;font-size:16px;">LETNet</th>
    <th style="text-align:center;font-size:16px;">LETNet2</th>
  </tr>
  <tr>
    <td align="center"><img src="assets/lk_heatmap.jpg" border="0" width="260"></td>
    <td align="center"><img src="assets/let_heatmap.jpg" border="0" width="260"></td>
    <td align="center"><img src="assets/ours_heatmap.jpg" border="0" width="260"></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/lk_frame000001_heatmap.jpg" border="0" width="260"></td>
    <td align="center"><img src="assets/let_frame000001_heatmap.jpg" border="0" width="260"></td>
    <td align="center"><img src="assets/ours_frame000001_heatmap.jpg" border="0" width="260"></td>
  </tr>
</table>


## Train

### Docker Build 

Download this project

```shell
git clone git@github.com:linyicheng1/LET-NET2.git
cd LET-NET2
```

Build the dockerfile 

```shell
docker build . -t letnet2
```

Waiting for Docker to build, this will take some time.


### Prepare the dataset

We train using [TartanAir](https://theairlab.org/tartanair-dataset/) data from the simulated environment.

According to the tutorial, we used the following commands to download the datasets, including both monocular and depth datasets.

```shell
pip install boto3 colorama minio
```

```shell
python download_training.py --output-dir OUTPUTDIR --rgb --only-left --depth --unzip
```

> The data is hosted on two servers located in the United States. By default, it downloads from AirLab data server. If you encounter any network issues, please try adding --cloudflare for an alternative source.

After downloading the monocular images and depth maps, we obtain the following file structure.

```shell
./office
  - Easy
    - P001
      - pose_left.txt
      - image_left
        - 000001_left.png
        - 000xxx_left.png
      - depth_left
        - 000001_left_depth.npy
        - 000xxx_left_depth.npy
    - P00x
  - Hard
    - P001
    - P00x
./seasidetown
./westerndesert
./amusement
./gascola
./ocean
./carwelding
./hospital
./abandonedfactory
./oldtown
./office2
./soulcity
./japanesealley
./abandonedfactory_night
./neighborhood
./seasonsforest
./seasonsforest_winter
./endofworld
```


### Training 

First, run Docker and transfer the code and training dataset into the Docker container.

```shell
docker run -it --gpus all -v CODE_DIR:/home/code  -v DATA_DIR:/home/data -p 2222:22 --name letnet2 letnet2:latest
```

Inside Docker, we can run the following command to start training.

In a new terminal 

```shell
docker exec -it letnet2 bash
cd /home/code/LET-NET2/
```

Run Training CMD

```shell
python3 train.py
```


## Interface 

### python

Install dependencies

```shell
pip install torch torchvision opencv-python numpy
```

Run Demo

```shell
cd interface/python
python3 demo.py -m /home/code/weight/letnet2.pth -i /home/code/interface/assets/nyu_snippet.mp4
```

### cpp (CPU)

#### Prerequisites 

- OpenCV (https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html)
- ncnn (https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux)

> Notes: After installing ncnn, you need to change the path in CMakeLists.txt

```
set(ncnn_DIR "<your_path>/install/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
```

#### Build 

```
mkdir build && cd build
cmake .. && make -j4
```

#### Run demo 

```
./demo ../weights/letnet_480x640.ncnn.param ../weights/letnet_480x640.ncnn.bin ../../assets/nyu_snippet.mp4 
```

### export && tensorrt 

Given that model export and GPU inference environments can be quite complex, we adopt Docker technology here to simplify this process.

#### Docker Build 

```shell
cd LET-NET2/interface
docker build . -t let2_interface
```

#### Run Docker 

```shell
docker run -it --gpus all -v ${CODE_DIR}/LET-NET2:/home/code -v ${DATA_DIR}/euroc/:/home/data/ let2_interface:latest 
```

#### Export models

```shell
cd /home/code/interface/python/
python3.10 export.py --model /home/code/weight/letnet2.pth --height 480 --width 640
```

You can adjust the image dimensions to export various models.


#### Tensorrt interface 

```shell
cd /home/code/interface/cpp_tensorrt/
mkdir build && cd build/
cmake .. && make -j4
./demo_trt ../weights/letnet_480x640.engine ../../assets/nyu_snippet.mp4
```



## VINS-Fusion Integration


### Docker Build 

```shell
cd LET-NET2/VINS
docker build . -t let_vins
```


### Run Docker 

```shell
docker run -it --gpus all -v ${CODE_DIR}/LET-NET2/VINS:/home/code/src -v ${DATA_DIR}/euroc/:/home/data/ --net=host --env ROS_MASTER_URI=http://localhost:11311 --env ROS_IP=$(hostname -I | awk '{print $1}') let_vins:latest 
```

my sample

```shell
docker run -it --gpus all -v /home/server/linyicheng/LETNET2/LET-NET2/VINS:/home/code/src -v /media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/euroc/:/home/data/ --net=host --env ROS_MASTER_URI=http://localhost:11311 --env ROS_IP=$(hostname -I | awk '{print $1}') let_vins:latest 
```

### Build LET-VINS 

```shell
cd /home/code/ && catkin_make
```

### Run LET-VINS 

```shell
source devel/setup.bash
rosrun vins vins_node src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml
```

play rosbag 


```shell
source /opt/ros/melodic/setup.bash
cd /home/data/
rosbag play MH_05_difficult.bag
```
