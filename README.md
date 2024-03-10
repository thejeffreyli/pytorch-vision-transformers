# All We Do Is SWIN: Evaluation Of Vision Transformers And Its Variants On Image Classification Tasks

By Jeffrey Li, Dontr (Dante) Lokitiyakul, Yash Nakadi, Thiti (Nob) Premrudeepreechacharn

Last Updated: December 19, 2023

## Project Summary

There has been a recent interest in applying Transformers to tasks in computer vision, such as image segmentation and image classification. In particular, Vision Transformers (ViT) and Hierarchical Vision Transformers via Shifted Windows were used to compare the performance of Convolutional Neural Networks (CNNs) in the image classification task on the CIFAR-100 dataset. With regularization, data augmentation, constructing hybrid models, and tuning hyperparameters, we found that the Swin architecture with increased number of Swin blocks outperforms all model and experimental configurations at 77.1\% test accuracy. This can be compared to the best performance of the CNN model (67\%) and the ViT (54.5\%). Given its performance and hierarchical structure, we also explored its potential as a basis unit for encoder and decoder in generative modeling, notably Variational Autoencoder.


## Content

- **cnn:** contains relevant code for CNN 
    - cnn.ipynb: training and validation procedures for CNN model

- **papers:** 
    - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](./papers/swin.pdf)
    - [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](./papers/train-vit.pdf)
    - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](./papers/vit.pdf)

- **report:** contains final report submission

- **swin:** contains relevant code for SWIN Transformer model
    - datasets.py: loads CIFAR100 data
    - engine.py: trains and evaluates functions used in main.py
    - main.py: passes arguments
    - swin.py: implementation of SWIN Transformer architecture 
    - utils.py: helper functions, including distributed helpers
    - vae.py: experiments with SWIN Transformer model as VAE
    - visualization_swin.ipynb: visualizing plots and experimental results

- **vit:** contains relevant code for ViT Model
    - vit_model.py: implementation of ViT architecture, contains training and validation procedures
    - vit-resnet34-hybrid.py: implementation of ViT-ResNet34 Hybrid architecture, contains training and validation procedures

## Program and System Requirements

We developed this primarily using AWS EC2, Python, PyTorch, and CUDA GPU.

Packages to Install:

```
pip install matplotlib
pip install torch
pip install timm
pip install tqdm
pip install PIL
pip install opencv-python
pip install numpy
pip install scikit-learn
pip install einops
pip install thop
pip install seaborn
pip install pandas

```

Dataset used: [CIFAR 100](https://www.cs.toronto.edu/~kriz/cifar.html). 

## PointNet Architecture

![PointNet](/assets/img/pointnet.jpg)

The classification network takes n points as input, applies input and feature transformations, and then aggregates point features by max pooling. The output is classification scores for k classes.

## PointNN Architecture

<div style="display: flex; justify-content: space-between;">
    <img src="/assets/img/pointnn01.png" alt="NPE" width="400" style="margin-right: 10px;">
    <img src="/assets/img/pointnn02.png" alt="PMB" width="400"/>
</div>

Non-Parametric Encoder (Left). Zhang et al. (2023) utilized trigonometric functions to encode raw Point Clouds points into high-dimensional vectors in PosE. The vectors then pass through a series of hierarchical non-parametric operations, namely FPS, k-NN, local geometric aggregation, and pooling, where they will be encoded into a global feature $f_G$. Point-Memory Bank (Right). The training set features are passed through the Point-Memory Bank outputting $F_{mem}$, which is later used to classify using similarity matching.


## PointNet Robustness Testing: Rotation

![rotation](/assets/img/pointcloud-rotation.png)


Sample point cloud demonstrating a chair undergoing rotations (left to right): 0 degrees, 5 degrees, 30 degrees, 45 degrees, 90 degrees. The model incorrectly classifies the chair starting at the 30 degrees rotation. 

## PointNet Robustness Testing: Data Corruption

![corruption](/assets/img/pointcloud-sampling.png)

Sample point cloud demonstrating a chair undergoing sampling (left to right): 10000 points, 7500 points, 5000 points, 2500 points, 1000 points, 500 points, 100 points. The model correctly classifies the chair at all different samplings.


## Results

### Accuracy and Loss Curves

![acc_curve](/assets/img/pointnetplotacc250.png)

![loss_curve](/assets/img/pointnetplotloss250.png)

PointNet Training and testing curves over 250 epochs.

### Robustness Testing for Best PointNet Model

| Number of Samples | Accuracy | Degree of Rotation | Accuracy |
|-------------------|----------|---------------------|----------|
| 10000             | 98.3%    | 0°                  | 98.3%    |
| 7500              | 98.3%    | 5°                  | 98.1%    |
| 5000              | 98.3%    | 30°                 | 73.6%    |
| 2500              | 98.1%    | 45°                 | 43.2%    |
| 1000              | 97.5%    | 90°                 | 25.3%    |
| 500               | 97.0%    |                     |          |
| 100               | 94.6%    |                     |          |


### Visualization of Point Cloud Data Encoding by PointNN FPS

![fps](/assets/img/fps.png)

Sample point clouds undergoing four FPS iterations (left to right): 10000 points, 5000 points, 2500 points, 1250 points, 625 points.

### Visualization of Point Cloud Data Encoding by PointNN k-NN

![knn](/assets/img/knn.png) 

Sample point clouds in the nth stage of the multistage hierarchy undergoing FPS and k-NN with k = 90 (from left to right): the point cloud before FPS, the point cloud after FPS, and k-NN where red indicates the nearest neighbors (cluster) of a selected point in the cloud (after FPS).


## Conclusion

Point cloud processing is important in several fields. In this project, we examined and assessed two different methods for classifying point clouds on a point cloud dataset. The PointNet architecture provides a way to learn global and local point features while also achieving permutation invariance, whereas the Point-NN network offers a way to capture features, like spatial patterns and structures, through a non-parametric approach. Through our experiments, we built and implemented both networks to understand the different mechanisms that allow them to perform classification tasks. Though PointNet achieves higher accuracy when evaluated on the test dataset, we acknowledge the ability for Point-NN to achieve reasonable accuracy with no training and substantially less computational resources. Additionally, we performed robustness tests, specifically data corruption and rotations, in order to evaluate PointNet's ability to classify objects despite changes to orientation and structure. From our tests, we found that PointNet is capable of recognizing global structures of objects despite missing points, but it suffers when large rotations are applied to the point clouds. Lastly, we visually analyzed the internal representations of the multistep hierarchical layers inside the non-parametric encoder to understand how Point-NN captures meaningful representations in point cloud data. 

We have several directions in which we can expand upon our existing work. Firstly, there are several relevant and more advanced architecture capable of processing point cloud data, while also performing other vision tasks, such as segmentation and object detection, that we can further explore. PointNet is a stepping stone to more exciting networks, such as PointNet++, DGCNN, and Point Transformers, in which several changes and design decisions have been made to account for the drawbacks of the original PointNet. Secondly, Point-NN's utility can be expanded upon as a plug-and-play module to boost existing learnable 3D models without further training. Applying Point-NN's ability to capture spatial representations will enhance the shortcomings to baseline models such as PointNet. Thirdly, acquiring access to greater computing resources will allow us to build upon our existing work through being able to keep and run the Transformation-Network as well as the full ModelNet40 point cloud dataset. 