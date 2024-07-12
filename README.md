<p align="center" width="100%">
<img src="images/IMDL_BenCo.png" alt="OSQ" style="width: 28%; min-width: 150px; display: block; margin: auto;">
</p>

# IMDL-BenCo:  Comprehensive Benchmark and Codebase for Image Manipulation Detection & Localization
<div align="center">
Xiaochen Ma†, Xuekang Zhu†, Lei Su†, Bo Du†, Zhuohang Jiang†, Bingkui Tong†,
Zeyu Lei†, Xinyu Yang†, Chi-Man Pun, Jiancheng Lv, Jizhe Zhou*
</div>  
<div align="center"><span style="font-size: smaller;">
<br>†: joint first author & equal contribution
*: corresponding author</br>
🏎️Special thanks to Dr. <a href="https://cs.scu.edu.cn/info/1359/17839.htm">Wentao Feng</a> for the workplace, computation power, and physical infrastructure support.</span>    
</div>  

******  

[![Powered by](https://img.shields.io/badge/Based_on-Pytorch-blue?logo=pytorch)](https://pytorch.org/) 
[![Arxiv](https://img.shields.io/badge/arXiv-2406.10580-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2406.10580)
[![Documents](https://img.shields.io/badge/Documents-Click_here-brightgreen?logo=read-the-docs)](https://scu-zjz.github.io/IMDLBenCo-doc/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/imdlbenco?label=PyPI%20Downloads&logo=pypi&logoColor=white&color=blue)](https://pypi.org/project/IMDLBenCo/)
![last commit](https://img.shields.io/github/last-commit/scu-zjz/IMDLBenCo)
![GitHub](https://img.shields.io/github/license/scu-zjz/IMDLBenCo?logo=license)

<!----
[![Ask Me Anything!](https://img.shields.io/badge/Official%20-Yes-1abc9c.svg)](https://GitHub.com/scu-zjz/) 
---->

## Overview
☑️**Welcome to IMDL-BenCo, the first comprehensive IMDL benchmark and modular codebase.**    
- This codebase is under long-term maintenance and updating. New features, extra baseline/sota models, and bug fixes will be continuously involved. You can find the corresponding plan here shortly.
- This repo decomposes the IMDL framework into **standardized, reusable components and revises the model construction pipeline**, improving coding efficiency and customization flexibility.
- This repo **fully implements or incorporates training code for state-of-the-art models** to establish a comprehensive IMDL benchmark.
- Cite and star if you feel helpful. This will encourage us a lot 🥰.   

☑️**About the Developers:**  
- IMDL-BenCo's project leader/supervisor is Associate Professor 🏀[_Jizhe Zhou_ (周吉喆)](https://knightzjz.github.io/), Sichuan University🇨🇳.  
- IMDL-BenCo's codebase designer and coding leader is Research Assitant [_Xiaochen Ma_ (马晓晨)](https://me.xiaochen.world/), Sichuan University🇨🇳.  
- IMDL-BenCo is jointly sponsored and advised by Prof. _Jiancheng LV_ (吕建成), Sichuan University 🐼, and Prof. _Chi-Man PUN_ (潘治文), University of Macau 🇲🇴, through the [Research Center of Machine Learning and Industry Intelligence, China MOE](https://center.dicalab.cn/) platform.  

**Important! The current documentation and tutorials are not complete. This is a project that requires a lot of manpower, and we will do our best to complete it as quickly as possible. 
Currently, you can use the demo following the brief tutorial below.**
![](./images/IMDLBenCo_overview.png)

## Features under developing
This repository has completed training, testing, robustness testing, Grad-CAM, and other functionalities for mainstream models.

However, more features are currently in testing for improved user experience. Updates will be rolled out frequently. Stay tuned!

- [ ] Install and download via PyPI
   - [x] You can experience on test PyPI now! 
- [x] Based on command line invocation, similar to `conda` in Anaconda.
   - [x] Dynamically create all training scripts to support personalized modifications.

- [ ] Information library, downloading, and re-management of IMDL datasets.
- [ ] Support for Weight & Bias visualization.

## Quick Experience about our Modern Paradigm (New!)
We are rapidly advancing the testing of the CLI and other features. In the meantime, you can experience the development paradigm designed by IMDL-BenCo via `PyPI`. We strive to provide users with enough personalized options while maximizing the efficiency, accuracy, and fairness of completing IML experiments.

```shell
pip install imdlbenco==0.1.10
```

After installation, you can create a clean working directory and execute the following command. IMDL-BenCo will automatically generate some files in the current directory as a minimal demo for you to experience our paradigm.

```shell
benco init
```
If everything works correctly, you can find following files:

```
.
├── mymodel.py
├── README-IMDLBenCo.md
├── test_datasets.json
├── test_mymodel.sh
├── test.py
├── test_robust_mymodel.sh
├── test_robust.py
├── train_mymodel.sh
└── train.py
```
At this point, you can complete the testing process by following the guidance in `README-IMDLBenCo.md`. Alternatively, you can directly refer to the [`README-IMDLBenCo.md`](https://github.com/scu-zjz/IMDLBenCo/blob/dev-cmd/IMDLBenCo/statics/base/README-IMDLBenCo.md) in this repository to understand the relevant paradigm.

> Currently, it may be necessary to correctly configure the dataset before running. Please refer to the configuration below temporarily, or follow the dataset organization method in [IML-ViT](https://github.com/SunnyHaze/IML-ViT) to configure it. Then, passing the corresponding location into the shell script will be ok.


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
Our design paradigm aims for the majority of customization for new models (including specific models and their respective losses) to occur within the model_zoo. Therefore, we have adopted a special design paradigm to interface with other modules. It includes the following features:

- Loss functions are defined in `__init__` and computed within `forward()`.
- The parameter list of `forward()` must consist of fixed keys to correspond to the input of required information such as `image`, `mask`, and so forth. Additional types of information can be generated via post_func and their respective fields, accepted through corresponding parameters with the same names in `forward().`
- The return value of the `forward()` function is a well-organized dictionary containing the following information as an example:
```python
  # -----------------------------------------
  output_dict = {
      # loss for backward
      "backward_loss": combined_loss,
      # predicted mask, will calculate for metrics automatically
      "pred_mask": mask_pred,
      # predicted binaray label, will calculate for metrics automatically
      "pred_label": None,

      # ----values below is for visualization----
      # automatically visualize with the key-value pairs
      "visual_loss": {
        # customized float for visualize, the key will shown as the figure name. Any number of keys and any str can be added as key.
          "predict_loss": predict_loss,
          "edge_loss": edge_loss,
          "combined_loss": combined_loss
      },

      "visual_image": {
        # customized tensor for visualize, the key will shown as the figure name. Any number of keys and any str can be added as key.
          "pred_mask": mask_pred,
          "edge_mask": edge_mask
  }
      # -----------------------------------------
```

Following this format, it is convenient for the framework to backpropagate the corresponding loss, compute final metrics using masks, and visualize any other scalars and tensors to observe the training process.



## Citation
If you find our work valuable and it has contributed to your research or projects, we kindly request that you cite our paper. Your recognition is a driving force for our continuous improvement and innovation🤗.
```
@misc{ma2024imdlbenco,
    title={IMDL-BenCo: A Comprehensive Benchmark and Codebase for Image Manipulation Detection & Localization},
    author={Xiaochen Ma and Xuekang Zhu and Lei Su and Bo Du and Zhuohang Jiang and Bingkui Tong and Zeyu Lei and Xinyu Yang and Chi-Man Pun and Jiancheng Lv and Jizhe Zhou},
    year={2024},
    eprint={2406.10580},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

**************
<div align="center">
<a href="https://info.flagcounter.com/H5vw"><img src="https://s11.flagcounter.com/count2/H5vw/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_12/viewers_0/labels_0/pageviews_1/flags_0/percent_0/" alt="Flag Counter" border="0"></a></div>
