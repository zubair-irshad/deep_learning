My solutions (with explanations) to [CS-7643 Deep Leanring](https://www.cc.gatech.edu/classes/AY2020/cs7643_spring/) course taught in Spring'2020. This repo is divided into 3 major components:


Vanilla Neural Net & Convnet
----------------------------

**Implementing a vanilla Neural Network from scratch**

- Implementation of a 2 layer neural net from scratch in python (Yes! this project implements backprop, loss functions and training loop all without using an external library such as pytorch/tensorflow. Library implementation comes later). Implemented 2 layer neural net for [image classification on CIFA-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) 

**Implementing a vanilla Convolution Neural Network from scratch**

- Implementation of a  2 layer Conv-net from scratch in python. This project implements the convolution functions, backward pass and training loop of a simple 2 layer conv-net from scratch. Inspiration was taken from the amazing [CS source](https://cs231n.github.io/convolutional-networks/) taught by Andrej Karpathy.

**CIFAR-10 classifier**

- Implementation of Linear Classifier and Convulutional Neural Net based classifier for [image classification on CIFA-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) . This project also implements a [Trasnfer Learning](https://en.wikipedia.org/wiki/Transfer_learning) solution to achieve a 95% accuracy on CIFAR-10 dataset.

Visualizations of Convolution Neural Networks
--------------------------------------------

**Style Trasnfer using convolutional neural nets**

- Implementation of [Style Transfer algorithm](https://ieeexplore.ieee.org/document/7780634) to generate cool styles for random images using conv-nets in pytorch


**Saliency maps, Grad-CAM and Guided Backprop using Pytorch and Captum**

- Implementation of [Saliency maps](https://arxiv.org/abs/1312.6034),[Grad-CAM](https://arxiv.org/abs/1610.02391) and [Guided Backprop](https://arxiv.org/abs/1610.02391) on pytorch from scratch and using [Facebook's Captum Library](https://captum.ai/)

Image captioning using Recurrent Neural Nets
--------------------------------------------

**Image captioning using Vanilla RNN**

Implementation of a vanilla Recurrent Neural Network from scratch in python to generate captions for unseen images. Training was carried on on [COCO dataset](http://cocodataset.org/#home)


**Image captioning using Vanilla LSTM**

Implementation of a vanilla Long short term Memory Network from scratch in python to generate captions for unseen images. Training was carried on on [COCO dataset](http://cocodataset.org/#home)

Final Project
--------------
**Supervised Learning Benchmarks For Embodied Visual Navigation In Habitat**

[Project report can be access here](https://zubairirshad.com/portfolio/supervised-learning-benchmarks-for-embodied-visual-navigation-in-habitat/)

[Github implementation of the project is available here](https://github.com/zubair-irshad/habitat_imitation_learning)

