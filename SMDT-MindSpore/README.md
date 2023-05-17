# SMDT-MindSpore
## will be updated soon
**[SMDT: Cross-View Geo-Localization with Image Alignment and Transformer.]()**  [ICME 2022]().

### MindSpore Version
The repository offers the main implementation of our paper in MindSpore.

#### Dependencies
tqdm==4.9.0<br>
matplotlib==2.1.2<br>
mindspore==1.6.1 [Install](https://www.mindspore.cn/install)

### PyTorch Version
The PyTorch version is available at [https://github.com/TianXiaoYang-txy/SMDT_PyTorch](https://github.com/TianXiaoYang-txy/SMDT_PyTorch)

## Contents
  - [Abstract](#Abstract)
  - [Datasets](#Datasets)
  - [Image Alignment](#Image_Alignmentn)
  - [Transformer](#Transformer)
  - [X2MindSpore](#X2MindSpore)
  - [Results](#Results)
  - [Acknowledgments](#Acknowledgments)

## Abstract
The goal of cross-view geo-localization is to determine the location of a given ground image by matching with aerial images. However, existing methods ignore the variability of scenes, additional information and spatial correspondence of covisibility and non convisibility areas in ground-aerial image pairs. In this context, we propose a cross-view matching method called SMDT with image alignment and Transformer. First, we utilize semantic segmentation technique to segment different areas. Then, we convert the vertical view of aerial images to front view by mixing polar mapping and perspective mapping. Next, we simultaneously train dual conditional generative adversarial nets by taking the semantic segmentation images and converted images as input to synthesize the aerial image with ground view style. These steps are collectively referred to as image alignment. Last, we use Transformer to explicitly utilize the properties of self-attention. Experiments show that our SMDT method is superior to the existing ground-to-aerial cross-view methods.

## Dataset

* CVUSA：[https://github.com/viibridges/crossnet](https://github.com/viibridges/crossnet)
* CVACT：[https://github.com/Liumouliu/OriCNN](https://github.com/Liumouliu/OriCNN)

## Image Alignment

Please refer to the [Image_Alignment](https://github.com/TianXiaoYang-txy/SMDT_MindSpore/tree/main/Image_Alignment) folder for more details.

#### Semantic Segmentation
<center>
<img src='./imgs/fig1.jpg' width=1000>
</center>

#### Mixed Perspective-Polar Mapping
<center>
<img src='./imgs/fig2.jpg' width=400>
</center>

#### Dual Conditional Generative Adversarial Nets
<center>
<img src='./imgs/fig3.jpg' width=400>
</center>

## Transformer

Please refer to the [Transformer](https://github.com/TianXiaoYang-txy/SMDT_MindSpore/tree/main/Transformer) folder for more details.

#### Transformer for Cross-View Geo-localization
<center>
<img src='./imgs/fig4.jpg' width=400>
</center>

## X2MindSpore
The script conversion tool [X2MindSpore](https://support.huaweicloud.com/devtool-cann51RC1alpha3/atlasfmkt_16_0002.html) is used to directly convert PyTorch into MindSpore version.

## Results

#### Results on CVUSA
<center>
<img src='./imgs/tab1.jpg' width=400>
</center>

#### Results on CVACT
<center>
<img src='./imgs/tab2.jpg' width=400>
</center>

## Acknowledgment
This source code of Image Alignment is inspired by [PanoGAN](https://github.com/sswuai/PanoGAN), the source code of Transformer is inspired by [Polar-EgoTR](https://github.com/yanghongji2007/cross_view_localization_L2LTR)# SMDT_MindSpore
# SMDT_MindSpore
