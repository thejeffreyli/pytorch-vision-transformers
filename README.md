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

## CNN Architecture

![cnn](/assets/img/cnn_architecture.png)
<img src="/assets/img/cnn.png" alt="cnn" width="300"/>

xxx
## ViT Architecture

![vit](/assets/img/vit.png)

## SWIN ViT Architecture

![swin](/assets/img/swin.png)


## Results

### CNN Results

| Dataset Type     | Our CNN | ResNet34 |
|-------------------|---------|----------|
| Augmentation      | 67%     | 62%      |
| No Augmentation   | 54%     | 58%      |

Test Accuracy of CNN-based Models

![cnn_loss](/assets/img/cnn_loss.png)

![cnn_acc](/assets/img/cnn_accuracy_all_models.png)




### ViT Results

| Trial | Augment | Hybrid | MLP-Head | Patch Size | Embed Dim | Num Layers | Num Heads | Test Acc |
|-------|---------|--------|----------|------------|-----------|------------|-----------|----------|
| A     | No      | No     | Original | 16         | 768       | 12         | 12        | 27.7%    |
| B     | No      | No     | Original | 4          | 768       | 12         | 12        | 31.7%    |
| C     | No      | No     | Original | 4          | 512       | 12         | 16        | 31.6%    |
| D     | Yes     | No     | Modified | 4          | 512       | 8          | 16        | 31.9%    |
| E     | Yes     | Yes    | Original | 7          | 512       | 8          | 16        | 47.7%    |
| F     | Yes     | Yes    | Modified | 7          | 512       | 12         | 16        | 54.5%    |


### SWIN ViT Results

| Name                        | Configuration        | Activation function | Test Acc |
|-----------------------------|----------------------|----------------------|----------|
| GELU Original / Original    | `[2, 2, 6, 2]`       | GELU                 | 76.8%    |
| Simple                      | `[2, 2, 2, 2]`       | GELU                 | 76.38%   |
| Complex                     | `[2, 2, 10, 2]`      | GELU                 | 77.1%    |
| GELU                        | `[2, 2, 2, 2]`       | GELU                 | 75.2%    |
| ReLU                        | `[2, 2, 2, 2]`       | ReLU                 | 73.4%    |
| Swish                       | `[2, 2, 2, 2]`       | Swish                | 73.7%    |


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
