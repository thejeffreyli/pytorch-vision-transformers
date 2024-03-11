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

<img src="/assets/img/cnn.png" alt="cnn" width="500"/>

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

Test Accuracy of CNN-based Models.

<div style="display: flex; justify-content: space-between;">
    <img src="/assets/img/cnn_loss.png" alt="cnn_loss" width="400" style="margin-right: 10px;">
    <img src="/assets/img/cnn_accuracy_all_models.png" alt="cnn_accuracy_all_models" width="400">
</div>

(Left) Training loss of CNN-based models. (Right) Testing accuracy of CNN-based models. 


### ViT Results

| Trial | Augment | Hybrid | MLP-Head | Patch Size | Embed Dim | Num Layers | Num Heads | Test Acc |
|-------|---------|--------|----------|------------|-----------|------------|-----------|----------|
| A     | No      | No     | Original | 16         | 768       | 12         | 12        | 27.7%    |
| B     | No      | No     | Original | 4          | 768       | 12         | 12        | 31.7%    |
| C     | No      | No     | Original | 4          | 512       | 12         | 16        | 31.6%    |
| D     | Yes     | No     | Modified | 4          | 512       | 8          | 16        | 31.9%    |
| E     | Yes     | Yes    | Original | 7          | 512       | 8          | 16        | 47.7%    |
| F     | Yes     | Yes    | Modified | 7          | 512       | 12         | 16        | 54.5%    |

<div style="display: flex; justify-content: space-between;">
    <img src="/assets/img/vit_train_acc.png" alt="vit_train_acc" width="400" style="margin-right: 10px;">
    <img src="/assets/img/vit_test_acc.png" alt="vit_test_acc" width="400">
</div>

(Left) Training accuracy of ViT-based models. (Right) Testing accuracy of ViT-based models.

<div style="display: flex; justify-content: space-between;">
    <img src="/assets/img/vit_train_loss.png" alt="vit_train_loss" width="400" style="margin-right: 10px;">
    <img src="/assets/img/vit_test_loss.png" alt="vit_test_loss" width="400">
</div>

(Left) Training loss of ViT-based models. (Right) Testing loss of ViT-based models.

### SWIN ViT Results

| Name                        | Configuration        | Activation function | Test Acc |
|-----------------------------|----------------------|----------------------|----------|
| GELU Original / Original    | `[2, 2, 6, 2]`       | GELU                 | 76.8%    |
| Simple                      | `[2, 2, 2, 2]`       | GELU                 | 76.38%   |
| Complex                     | `[2, 2, 10, 2]`      | GELU                 | 77.1%    |
| GELU                        | `[2, 2, 2, 2]`       | GELU                 | 75.2%    |
| ReLU                        | `[2, 2, 2, 2]`       | ReLU                 | 73.4%    |
| Swish                       | `[2, 2, 2, 2]`       | Swish                | 73.7%    |

![SWIN_train_perf_diff_complexity](/assets/img/SWIN_train_perf_diff_complexity.png)

![SWIN_test_perf_diff_complexity](/assets/img/SWIN_test_perf_diff_complexity.png)

![SWIN_train_perf_diff_activation](/assets/img/SWIN_train_perf_diff_activation.png)

![SWIN_test_perf_diff_activation](/assets/img/SWIN_test_perf_diff_activation.png)



## PointNet Robustness Testing: Data Corruption

xxx


## Results


## Conclusion
