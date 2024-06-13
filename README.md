<p align="center" width="100%">
<img src="images/IMDL_BenCo.png" alt="OSQ" style="width: 28%; min-width: 150px; display: block; margin: auto;">
</p>

# IMDLBenCo:  Comprehensive Benchmark and Codebase for Image Manipulation Detection & Localization

Xiaochen Ma†, Xuekang Zhu†, Lei Su†, Bo Du†, Zhuohang Jiang†, Bingkui Tong1†,
Zeyu Lei†, Xinyu Yang†, Chi-Man Pun, Jiancheng Lv, Jizhe Zhou*

<div style="text-align: center;"><span style="font-size: smaller;">
†: joint first author & equal contribution
*: corresponding author.</span>
</div>


![Powered by](https://img.shields.io/badge/Based_on-Pytorch-blue?logo=pytorch) 
![last commit](https://img.shields.io/github/last-commit/scu-zjz/IMDLBenCo)
![GitHub](https://img.shields.io/github/license/scu-zjz/IMDLBenCo?logo=license)
![](https://img.shields.io/github/repo-size/scu-zjz/IMDLBenCo?color=green)
![](https://img.shields.io/github/stars/scu-zjz/IMDLBenCo/)
[![Ask Me Anything!](https://img.shields.io/badge/Official%20-Yes-1abc9c.svg)](https://GitHub.com/scu-zjz/) 


************
## Features under developing
This repository has completed training, testing, robustness testing, Grad-CAM, and other functionalities for mainstream models.

However, more features are currently in testing for improved user experience. Updates will be rolled out frequently. Stay tuned!

- [ ] Install and download via PyPI
- [ ] Based on command line invocation, similar to `conda` in Anaconda.
   - [ ] Dynamically create all training scripts to support personalized modifications.

- [ ] Information library, downloading, and re-management of IMDL datasets.
- [ ] Support for Weight & Bias visualization.

## Overview
Welcome to IMDL-BenCo, the first comprehensive IMDL benchmark and modular codebase. 

This repo:
- decomposes the IMDL framework into standardized, reusable components and revises the model construction pipeline, improving coding efficiency and customization flexibility;
- fully implements or incorporates training code for state-of-the-art models to establish a comprehensive IMDL benchmark;

![](./images/IMDLBenCo_overview.png)


## Quick start
### Prepare environments
Currently, you can create a PyTorch environment and run the following command to try our repo.
```shell
git clone https://github.com/scu-zjz/IMDLBenCo.git
cd IMDLBenCo
pip install -r requirements.txt
```

### Prepare IML Datasets
- We defined three types of Dataset class
  - `JsonDataset`, which gets input image and corresponding ground truth from a JSON file with a protocol like this:
    ```
    [
        [
          "/Dataset/CASIAv2/Tp/Tp_D_NRN_S_N_arc00013_sec00045_11700.jpg",
          "/Dataset/CASIAv2/Gt/Tp_D_NRN_S_N_arc00013_sec00045_11700_gt.png"
        ],
        ......
        [
          "/Dataset/CASIAv2/Au/Au_nat_30198.jpg",
          "Negative"
        ],
        ......
    ]
    ```
    where "Negative" represents a totally black ground truth that doesn't need a path (all authentic)
  - `ManiDataset` which loads images and ground truth pairs automatically from a directory having sub-directories named `Tp` (for input images) and `Gt` (for ground truths). This class will generate the pairs using the sorted `os.listdir()` function. You can take [this folder](https://github.com/SunnyHaze/IML-ViT/tree/main/images/sample_iml_dataset) as an example.
  - `BalancedDataset` is a class used to manage large datasets according to the training method of [CAT-Net](https://github.com/mjkwon2021/CAT-Net). It reads an input file as [`./runs/balanced_dataset.json`](./runs/balanced_dataset.json), which contains types of datasets and corresponding paths. Then, for each epoch, it randomly samples over 1800 images from each dataset, achieving uniform sampling among datasets with various sizes.

### Training
#### Prepare pre-trained weights (if needed)
Some models like TruFor may need pre-trained weights. Thus you need to download them in advance. You can check the guidance to download the weights in each folder under the `./IMDLBenCo/model_zoo` for the model. For example, the guidance for TruFor is under [`IMDLBenCo\model_zoo\trufor\README.md`](IMDLBenCo\model_zoo\trufor\README.md)

#### Run shell script
You can achieve customized training by modifying the dataset path and various parameters. For specific meanings of these parameters, please use python ./IMDLBenco/training_scripts/train.py -h to check.

By default, all provided scrips are called as follows:
```
sh ./runs/demo_train_iml_vit.sh
```

#### Visualize the loss & metrics & figures
Now, you can call a Tensorboard to visualize the training results by a browser.
```
tensorboard --logdir ./
```

### Customize your own model
我们的设计范式希望绝大部分对于新模型的自定义（包括具体的模型和相应的loss）只发生在model_zoo中，所以我们采用了一套特殊的设计范式来对接其他模块。包含如下特征：
- 损失函数在`__init__`中定义，并在`forward()`中被调用计算
- `forward()`的形参列表必须由固定的key组成，以对应传入相应的image mask等等模型所需的信息，可以通过post_func生成更多需要的信息类型和相应的字段，并通过`forward()`函数中的同名形参进行接受。
- `forward()`函数的返回值是一个组织好的字典，包含如下信息：


## Citation
Coming soon

**************
<div align="center">
<a href="https://info.flagcounter.com/H5vw"><img src="https://s11.flagcounter.com/count2/H5vw/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_12/viewers_0/labels_0/pageviews_1/flags_0/percent_0/" alt="Flag Counter" border="0"></a></div>
