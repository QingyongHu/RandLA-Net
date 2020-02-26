[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/191111236/semantic-segmentation-on-semantic3d)](https://paperswithcode.com/sota/semantic-segmentation-on-semantic3d?p=191111236)

# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds

[Qingyong Hu](https://www.cs.ox.ac.uk/people/qingyong.hu/), [Bo Yang](https://yang7879.github.io/), [Linhai Xie](https://www.cs.ox.ac.uk/people/linhai.xie/), [Stefano Rosa](https://www.cs.ox.ac.uk/people/stefano.rosa/), [Yulan Guo](http://yulanguo.me/), [Zhihua Wang](https://www.cs.ox.ac.uk/people/zhihua.wang/), [Niki Trigoni](https://www.cs.ox.ac.uk/people/niki.trigoni/), [Andrew Markham](https://www.cs.ox.ac.uk/people/andrew.markham/). [arXiv:1911.11236](https://arxiv.org/abs/1911.11236), 2019.
RandLA-Net in Tensorflow, **coming soon**


Introduction
-------------------

This repository contains the implementation of **RandLA-Net**, a simple and efficient neural architecture for semantic segmentation of large-scale 3D point clouds.

The following figure shows the basic building block of our RandLA-Net:

<p align="center"> <img src="figs/Fig3.png" width="100%"> </p>

## Qualitative Results

### S3DIS

| ![2](./figs/S3DIS_area2.gif)   | ![z](./figs/S3DIS_area3.gif) |
| ------------------------------ | ---------------------------- |


### Semantic3D

| ![z](./figs/Semantic3D-1.gif)    | ![z](./figs/Semantic3D-2.gif)   |
| -------------------------------- | ------------------------------- |


### SemanticKITTI

![zzz](./figs/SemanticKITTI-2.gif)    

## Quantitative Results

### Semantic3D

Quantitative results of different approaches on Semantic3D (reduced-8). Only the recent published approaches are compared. Accessed on 15 November 2019.

![a](./figs/Semantic3D_table.png)

### SemanticKITTI

Quantitative results of different approaches on SemanticKITTI dataset.

![s](./figs/SemanticKITTI_table.png)

### S3DIS

Quantitative results of different approaches on S3DIS dataset.

<p align="center"> <img src="./figs/S3DIS_table.png" width="50%"> </p>

### Demo
-------------------

<p align="center"> <a href="https://youtu.be/Ar3eY_lwzMk"><img src="./figs/demo_cover.png" width="50%"></a> </p>


### Citation
If you find our work useful in your research, please consider citing:

	@article{hu2019randla,
	  title={RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds},
	  author={Hu, Qingyong and Yang, Bo and Xie, Linhai and Rosa, Stefano and Guo, Yulan and Wang, Zhihua and Trigoni, Niki and Markham, Andrew},
	  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
	  year={2020}
	}


