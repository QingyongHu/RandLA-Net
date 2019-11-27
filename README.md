# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds
Qingyong Hu, Bo Yang, Linhai Xie, Stefano Rosa, Yulan Guo, Zhihua Wang, Niki Trigoni, Andrew Markham. [arXiv:1911.11236](https://arxiv.org/abs/1911.11236), 2019.
RandLA-Net in Tensorflow, coming soon

Introduction
-------------------

This repository contains the implementation of **RandLA-Net**, a simple and efficient neural architecture for semantic segmentation of large-scale 3D point clouds.

The following figure shows the basic building block of our RandLA-Net:

<p align="center"> <img src="figs/Fig3.png" width="100%"> </p>


### Semantic3D

Quantitative results of different approaches on Semantic3D (reduced-8). Only the recent published approaches are compared.Accessed on 15 November 2019
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



