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

- **swin:** contains relevant code for Swin Transformer model
    - datasets.py: loads CIFAR100 data
    - engine.py: trains and evaluates functions used in main.py
    - main.py: passes arguments
    - swin.py: implementation of Swin Transformer architecture 
    - utils.py: helper functions, including distributed helpers
    - vae.py: experiments with Swin Transformer model as VAE
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

### Swin ViT Classification Results

| Name                        | Configuration        | Activation function | Test Acc |
|-----------------------------|----------------------|----------------------|----------|
| GELU Original / Original    | `[2, 2, 6, 2]`       | GELU                 | 76.8%    |
| Simple                      | `[2, 2, 2, 2]`       | GELU                 | 76.38%   |
| Complex                     | `[2, 2, 10, 2]`      | GELU                 | 77.1%    |
| GELU                        | `[2, 2, 2, 2]`       | GELU                 | 75.2%    |
| ReLU                        | `[2, 2, 2, 2]`       | ReLU                 | 73.4%    |
| Swish                       | `[2, 2, 2, 2]`       | Swish                | 73.7%    |

![SWIN_train_perf_diff_complexity](/assets/img/SWIN_train_perf_diff_complexity.png)

omparison of different Swin architecture variants based on complexity via number of Swin blocks for train set.

![SWIN_test_perf_diff_complexity](/assets/img/SWIN_test_perf_diff_complexity.png)

Comparison of different Swin architecture variants based on complexity via number of Swin  blocks for test set.

![SWIN_train_perf_diff_activation](/assets/img/SWIN_train_perf_diff_activation.png)

Comparison of different activation functions used in Swin architecture with additional 2-layer MLP for train set.

![SWIN_test_perf_diff_activation](/assets/img/SWIN_test_perf_diff_activation.png)

Comparison of different activation functions used in Swin architecture with additional 2-layer MLP for test set.

## Swin ViT vs Variational Autoencoder (VAE) 

While performing a literature review on Swin architecture, we realized that the Swin architecture utilizes a hierarchical feature map such that the height and width dimensions of the images are reduced by two-fold after every step due to the Rearrange tool and moves them into the latent dimension $C$, multiplying that by four-fold. We used that observation, combining with the implementation of the Variational Autoencoder (VAE) from the class lecture and homework, along with the implementation from GitHub repository, to build an encoder-decoder model using the Swin architecture stages as the basis of the model, instead of the fully-connected neural networks or linear layers, which we coined the architecture as Swin-VAE. We approached this idea by mirroring the pattern of Swin architecture to upsample the height and width dimensions two-fold after each stage for the decoder part, with the final output being an image. Specifically, for the network configuration of [2,2,6,2], we will have a total of 8 stages with the configuration as [2,2,6,2,2,6,2,2]. The evaluation process using the sum of reconstruction loss and KL Divergence loss uses the same implementation as that of the VAE, which we also implemented using the inspiration from the GitHub repository VAE-tutorial. In addition, since this model creates more overhead, we reduced the batch size to 128 and the number of epochs to 50 for both Swin-VAE and VAE for the model to be trainable with the available resources.

![SWIN_vae_output](/assets/img/SWIN_vae_output.png)

Output visualizations for VAE using Swin stages as the basis units and VAE using linear layers as the basis units. 

![SWIN_recon_kld_loss_vae](/assets/img/SWIN_recon_kld_loss_vae.png)

Original VAE (VAE) and Swin-VAE (Swin) Reconstruction and KL Divergence loss versus epochs.

<img src="/assets/img/SWIN_loss_vae.png" alt="SWIN_loss_vae" width="400">

Original VAE (VAE) and Swin-VAE (Swin) Total loss versus epochs.

## Conclusion

We evaluated the usability of Transformers-based architectures for vision tasks (i.e., image classification on the CIFAR-100 dataset) such that the ViT fails to outperform the CNN, which is the current primary choice for computer vision problems. Vision Transformers lack the inherent inductive biases in CNNs, and the we acknowledge the improvement in performance following the addition of a ResNet34 backbone. 

As Swin architecture outperforms the other models, though with a heavy computational complexity due to the requirement of calculating the local attention windows, we would like to perform the experiments in the same vein with what we did with Swin\_T but for Swin\_L (i.e., configuration = [2,2,18,2], $C = 192$), which we fail to do so due to computational resources limitations. Moreover, we would like to explore combining Swin with models such as ResNet34 as it was done with CNN and ViT, even though it is worth noting that Swin already has some residual connections. Moreover, as discussed in section, which we only trained the models for 50 epochs, we would like to train them with more epochs (e.g., 300 epochs, which is akin to that of the image classification tasks) since 50 epochs did not result in the intuitive decoded image output for VAE, in addition to increasing the complexity of the original VAE by adding more linear layers.

Given more time and computational resources, we would like to perform more robust tests to understand the relationship between components in transformer architectures as well as study design implementations from other state-of-the-art models, such as Data-efficient Image Transformers (DEIT) and CrossViT for vision tasks. 