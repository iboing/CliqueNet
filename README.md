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

<div align=left><img src="https://raw.githubusercontent.com/iboing/CliqueNet/master/img/fig3.JPG" width="50%" height="50%">

Fig 3. Feature maps of Stage-I and Stage-II with the highest average activation in a pre-trained model.

## Comparision with state of the arts

<div align=left><img src="https://raw.githubusercontent.com/iboing/CliqueNet/master/img/tab2.JPG" width="95%" height="95%">

Tab 2. Error rates (%) on CIFAR-10, CIFAR-100, and SVHN without any data augmentation. In CliqueNets and DenseNets, k is the number of filters per layer, and T is the total number of layers in three blocks. “A, B, C” represents attentional transition, bottleneck and compression, respectively. The FLOPs of DenseNets are calculated by ourselves.

## Results on ImageNet

<div align=left><img src="https://raw.githubusercontent.com/iboing/CliqueNet/master/img/tab3.JPG" width="50%" height="50%">

Tab 3. Single crop error rates (%) on ImageNet. The * indicates the models without attentional transition.

