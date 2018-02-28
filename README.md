# CliqueNet


This repository is for the paper Convolutional Neural Networks with Alternately Updated Clique (to appear in CVPR 2018)

by Yibo Yang, Zhisheng Zhong, Tiancheng Shen, and [Zhouchen Lin](http://www.cis.pku.edu.cn/faculty/vision/zlin/zlin.htm)

### citation
If you find CliqueNet useful in your research, please consider citing:

	@inproceedings{yang18,
	 author={xxx},
	 title={Convolutional Neural Networks with Alternately Updated Clique},
	 journal={CVPR},
	 year={2018}
	}

### table of contents
- [Introduction](#introduction)
- [Usage](#usage)
- [Ablation study](#ablation-study)
- [Comparision with state of the arts](#comparision-with-state-of-the-arts)
- [Results on ImageNet](#results-on-imagenet)

## Introduction
CliqueNet is a newly proposed convolutional neural network architecture where any pair of layers in the same block are connected bilaterally (Fig 1). Any layer is both the input and output another one, and information flow can be maximized. During propagation, the layers are updated alternately (Tab1), so that each layer will always receive the feedback information from the layers that are updated more lately. We show that the refined features are more discriminative and lead to a better performance. On benchmark classification datasets including CIFAR-10, CIFAR-100, SVHN, and ILSVRC 2012, we achieve better or comparable results over state of the arts with fewer parameters.


<div align=left><img src="https://raw.githubusercontent.com/iboing/test-/master/img/fig1.JPG" width="40%" height="40%">

Fig 1. An illustration of a block with 4 layers. Node 0 denotes the input layer of this block.

<div align=left><img src="https://raw.githubusercontent.com/iboing/test-/master/img/fig2.JPG" width="50%" height="50%">

Tab 1. Updating rule in CliqueNet. "{}" denotes the concatenating operator.




## Usage



asdsdf

## Ablation study

asdsdf

## Comparision with state of the arts

asdsdf

## Results on ImageNet



### third title
#### fourth titile




add a figure

to appear in CVPR 2018



add a table

model-----|top1|top5
---|:--------:|:----------:
aa|33%|33%
gg|55%|66%


add a citation
