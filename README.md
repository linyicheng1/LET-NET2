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


## VINS-Fusion Integration



