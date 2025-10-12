# LET-NET2：An end-to-end lightweight CNN designed for sparse corner extraction and tracking.

LET-NET2 transforms the traditional LK optical flow process into a neural network layer by computing its derivatives, enabling end-to-end training of sparse optical flow. It preserves the original lightweight network structure and, during training on simulated datasets, autonomously learns capabilities such as edge orientation extraction and active enhancement of weak-texture regions. As a result, it demonstrates stronger tracking performance under challenging conditions, including dynamic lighting, weak textures, low illumination, and underwater blur.

## Effect of Tracking Region Comparison on Weak-Texture and Low-Light Performance


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


## Interface 


## VINS-Fusion Integration



