# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds
Qingyong Hu, Bo Yang, Linhai Xie, Stefano Rosa, Yulan Guo, Zhihua Wang, Niki Trigoni, Andrew Markham. [arXiv:1911.11236](https://arxiv.org/abs/1911.11236), 2019.
RandLA-Net in Tensorflow, coming soon

Introduction
-------------------

This repository contains the implementation of **RandLA-Net**, a simple and efficient neural architecture for semantic segmentation of large-scale 3D point clouds.

The following figure shows the basic building block of our RandLA-Net:

<p align="center"> <img src="figs/Fig3.png" width="100%"> </p>


### Semantic3D

Quantitative results of different approaches on Semantic3D (reduced-8). Only the recent published approaches are compared. Accessed on 15 November 2019.

|              | mIoU(%) | OA(%) | man-made. | natural. | high veg. | low veg. | buildings | hard scape | scanning art. | cars |
|-------------:|:-------:|:-----:|:---------:|:--------:|:---------:|:--------:|:---------:|:----------:|:-------------:|:----:|
|     SnapNet_ |   59.1  |  88.6 |    82.0   |   77.3   |    79.7   |   22.9   |    91.1   |    18.4    |      37.3     | 64.4 |
|     SegCloud |   61.3  |  88.1 |    83.9   |   66.0   |    86.0   |   40.5   |    91.1   |    30.9    |      27.5     | 64.3 |
|      RF_MSSF |   62.7  |  90.3 |    87.6   |   80.3   |    81.8   |   36.4   |    92.2   |    24.1    |      42.6     | 56.6 |
| MSDeepVoxNet |   65.3  |  88.4 |    83.0   |   67.2   |    83.8   |   36.7   |    92.4   |    31.3    |      50.0     | 78.2 |
|     ShellNet |   69.3  |  93.2 |    96.3   |   90.4   |    83.9   |   41.0   |    94.2   |    34.7    |      43.9     | 70.2 |
|       GACNet |   70.8  |  91.9 |    86.4   |   77.7   |    88.5   |   60.6   |    94.2   |    37.3    |      43.5     | 77.8 |
|          SPG |   73.2  |  94.0 |    97.4   |   92.6   |    87.9   |   44.0   |    83.2   |    31.0    |      63.5     | 76.2 |
|       KPConv |   74.6  |  92.9 |    90.9   |   82.2   |    84.2   |   47.9   |    94.9   |    40.0    |      77.3     | 79.9 |
|   **RandLA-Net** |   76.0  |  94.4 |    96.5   |   92.0   |    85.1   |   50.3   |    95.0   |    41.1    |      68.2     | 79.4 |
![example segmentation](./figs/semanticKitti.gif)

|    Methods    |    Size   | mIoU | Params (M) | road | sidewalk | parking | other-ground | building |  car | truck | bicycle | motorcycle | other-vehicle | vegetation | trunk  | terrain  | person | bicyclist | motorcyclist | fence  | pole  | traffic-sign |
|:-------------:|:---------:|:----:|:----------:|:----:|:--------:|:-------:|:------------:|:--------:|:----:|:-----:|:-------:|:----------:|:-------------:|:----------:|:------:|:--------:|:------:|:---------:|:------------:|:------:|:-----:|:------------:|
|    PointNet   |           | 14.6 |      3     | 61.6 |   35.7   |   15.8  |      1.4     |   41.4   | 46.3 |  0.1  |   1.3   |     0.3    |      0.8      |    31.0    |   4.6  |   17.6   |   0.2  |    0.2    |      0.0     |  12.9  |  2.4  |      3.7     |
|    SPGraph    |           | 17.4 |    0.25    | 45.0 |   28.5   |   0.6   |      0.6     |   64.3   | 49.3 |  0.1  |   0.2   |     0.2    |      0.8      |    48.9    |  27.2  |   24.6   |   0.3  |    2.7    |      0.1     |  20.8  |  15.9 |      0.8     |
|    SPLATNet   |  50K pts  | 18.4 |     0.8    | 64.6 |   39.1   |   0.4   |      0.0     |   58.3   | 58.2 |  0.0  |   0.0   |     0.0    |      0.0      |    71.1    |   9.9  |   19.3   |   0.0  |    0.0    |      0.0     |  23.1  |  5.6  |      0.0     |
|   PointNet++  |           | 20.1 |      6     | 72.0 |   41.8   |   18.7  |      5.6     |   62.3   | 53.7 |  0.9  |   1.9   |     0.2    |      0.2      |    46.5    |  13.8  |   30.0   |   0.9  |    1.0    |      0.0     |  16.9  |  6.0  |      8.9     |
|  TangentConv  |           | 40.9 |     0.4    | 83.9 |   63.9   |   33.4  |     15.4     |   83.4   | 90.8 |  15.2 |   2.7   |    16.5    |      12.1     |    79.5    |  49.3  |   58.1   |  23.0  |    28.4   |      8.1     |  49.0  |  35.8 |     28.5     |
|               |           |      |            |      |          |         |              |          |      |       |         |            |               |            |        |          |        |           |              |        |       |              |
|   SqueezeSeg  |           | 29.5 |      1     | 85.4 |   54.3   |   26.9  |      4.5     |   57.4   | 68.8 |  3.3  |   16.0  |     4.1    |      3.6      |    60.0    |  24.3  |   53.7   |  12.9  |    13.1   |      0.9     |  29.0  |  17.5 |     24.5     |
| SqueezeSeg V2 |           | 39.7 |      1     | 88.6 |   67.6   |   45.8  |     17.7     |   73.7   | 81.8 |  13.4 |   18.5  |    17.9    |      14.0     |    71.8    |  35.8  |   60.2   |  20.1  |    25.1   |      3.9     |  41.1  |  20.2 |     36.3     |
|  DarkNet21Seg | 64*2048px | 47.4 |     25     | 91.4 |   74.0   |   57.0  |     26.4     |   81.9   | 85.4 |  18.6 |   26.2  |    26.5    |      15.6     |    77.6    |  48.4  |   63.6   |  31.8  |    33.6   |      4.0     |  52.3  |  36.0 |     50.0     |
|  DarkNet53Seg |           | 49.9 |     50     | 91.8 |   74.6   |   64.8  |     27.9     |   84.1   | 86.4 |  25.5 |   24.5  |    32.7    |      22.6     |    78.3    |  50.1  |   64.0   |  36.2  |    33.6   |      4.7     |  55.0  |  38.9 |     52.2     |
|   **RandLA-Net** |  50K pts  | 50.3 |    0.95    | 90.4 |   67.9   |   56.9  |     15.5     |   81.1   | 94.0 |  42.7 |   19.8  |    21.4    |      38.7     |    78.3    |  60.3  |   59.0   |  47.5  |    48.8   |      4.6     |  49.7  |  44.2 |     38.1     |

![example segmentation](./figs/S3DIS_area2.gif)





