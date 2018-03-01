# CliqueNet


This repository is for the paper [Convolutional Neural Networks with Alternately Updated Clique](https://arxiv.org/abs/1802.10419) (to appear in CVPR 2018)

by Yibo Yang, Zhisheng Zhong, Tiancheng Shen, and [Zhouchen Lin](http://www.cis.pku.edu.cn/faculty/vision/zlin/zlin.htm)

### citation
If you find CliqueNet useful in your research, please consider citing:

	@inproceedings{yang18,
	 author={Yibo Yang and Zhisheng Zhong and Tiancheng Shen and Zhouchen Lin},
	 title={Convolutional Neural Networks with Alternately Updated Clique},
	 journal={arXiv preprint arXiv:1802.10419},
	 year={2018}
	}

### table of contents
- [Introduction](#introduction)
- [Usage](#usage)
- [Ablation experiments](#ablation-experiments)
- [Comparision with state of the arts](#comparision-with-state-of-the-arts)
- [Results on ImageNet](#results-on-imagenet)

## Introduction
CliqueNet is a newly proposed convolutional neural network architecture where any pair of layers in the same block are connected bilaterally (Fig 1). Any layer is both the input and output another one, and information flow can be maximized. During propagation, the layers are updated alternately (Tab 1), so that each layer will always receive the feedback information from the layers that are updated more lately. We show that the refined features are more discriminative and lead to a better performance. On benchmark classification datasets including CIFAR-10, CIFAR-100, SVHN, and ILSVRC 2012, we achieve better or comparable results over state of the arts with fewer parameters.


<div align=left><img src="https://raw.githubusercontent.com/iboing/CliqueNet/master/img/fig1.JPG" width="40%" height="40%">

Fig 1. An illustration of a block with 4 layers. Node 0 denotes the input layer of this block.

<div align=left><img src="https://raw.githubusercontent.com/iboing/CliqueNet/master/img/fig2.JPG" width="95%" height="95%">

Fig 2. An overview of a CliqueNet with three blocks. The input layer together with the Stage-II feature in each block are concatenated to be the block feature, and form part of the final representation after global pooling. The Stage-II feature passes through transition layers, which include a convolution and an average pooling to change map size, and then becomes the input of the next block.


<div align=left><img src="https://raw.githubusercontent.com/iboing/CliqueNet/master/img/tab1.JPG" width="50%" height="50%">

Tab 1. Updating rule in CliqueNet. "{}" denotes the concatenating operator.



## Usage

to update

## Ablation experiments

<div align=left><img src="https://raw.githubusercontent.com/iboing/CliqueNet/master/img/fig3.JPG" width="65%" height="65%">

|Model|block feature|transit|error(%)|
|---|---|---|---|
|CliqueNet(I+I)|X_0, Stage-I|Stage-I|6.64|
|CliqueNet(I+II)|X_0, Stage-I|Stage-II|6.1|
|CliqueNet(II+II)|X_0, Stage-II|Stage-II|5.76|


|Model|C10|C100|
|---|---|---|
|CliqueNet(X=0)|5.83|24.79|
|CliqueNet(X=2)|5.68|24.37|
|CliqueNet(X=4)|5.20
|CliqueNet(X=5)|5.12|23.98|

to update

demonstrate the effectiveness of CliqueNet's feature refinement.


## Comparision with state of the arts

|Model                               | FLOPs | Params | CIFAR-10 | CIFAR-100 | SVHN |
|------------------------------------| ------|--------| -------- |-----------|------|
|DenseNet (k = 12, T = 36)           | 0.53G | 1.0M   |  7.00    |  27.55    | 1.79 |
|DenseNet (k = 12, T = 96)           | 3.54G | 7.0M   |  5.77    |  23.79    | 1.67 |
|DenseNet (k = 24, T = 96)           | 13.78G| 27.2M  |  5.83    |  23.42    | 1.59 |
|CliqueNet (k = 36, T = 12)          | 0.91G | 0.94M  |  5.93    |  27.32    | 1.77 |
|CliqueNet (k = 64, T = 15)          | 4.21G | 4.49M  |  5.12    |  23.98    | 1.62 |
|CliqueNet (k = 80, T = 15)          | 6.45G | 6.94M  |  5.10    |  23.32    | 1.56 |
|CliqueNet (k = 80, T = 18)          | 9.45G | 10.14M |  5.06    |  23.14    | 1.51 |

Tab 2. Main results on CIFAR and SVHN without data augmentation.

|Model|Params|C10|C100|
|---|---|---|---|
|DenseNet(k=12,T=36)|1.02M|7.00|27.55|
|CliqueNet(k=12,T=36)|1.05M|5.79|26.85|
|||||
|DenseNet(k=24,T=18)|0.99M|7.13|27.70|
|CliqueNet(k=24,T=18)|0.99M|6.04|26.57|
|||||
|DenseNet(k=36,T=12)|0.96M|6.89|27.54|
|CliqueNet(k=36,T=12)|0.94M|5.93|27.32|
Tab 3.
to update

demonstrate the superiority of CliqueNet over DenseNet when there are no additional techniques(bottleneck, compression, etc.)

## Results on ImageNet

to update
