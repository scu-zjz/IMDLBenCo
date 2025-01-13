<p align="center" width="100%">
<img src="images/IMDL_BenCo.png" alt="OSQ" style="width: 28%; min-width: 150px; display: block; margin: auto;">
</p>

# [NeurIPS'24 Spotlight] IMDL-BenCo: Comprehensive Benchmark and Codebase for Image Manipulation Detection & Localization
<div align="center">

[Xiaochen Ma](https://ma.xiaochen.world/) †, Xuekang Zhu†, Lei Su†, Bo Du†, Zhuohang Jiang†, Bingkui Tong†,
Zeyu Lei†, Xinyu Yang†, Chi-Man Pun, Jiancheng Lv, [Jizhe Zhou](https://knightzjz.github.io/) *

</div>  
<div align="center"><span style="font-size: smaller;">
<br>†: joint first author & equal contribution
*: corresponding author</br>
🏎️Special thanks to Dr. <a href="https://cs.scu.edu.cn/info/1359/17839.htm">Wentao Feng</a> for the workplace, computation power, and physical infrastructure support.</span>    
</div>  

******
<!----
[![Powered by](https://img.shields.io/badge/Based_on-Pytorch-blue?logo=pytorch)](https://pytorch.org/) 
---->
[![Arxiv](https://img.shields.io/badge/arXiv-2406.10580-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2406.10580)
[![Documents](https://img.shields.io/badge/Documents-Click_here-brightgreen?logo=read-the-docs)](https://scu-zjz.github.io/IMDLBenCo-doc/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/imdlbenco?label=PyPI%20Downloads&logo=pypi&logoColor=white&color=blue)](https://pypistats.org/packages/imdlbenco)
[![pypi version](https://img.shields.io/pypi/v/imdlbenco?logo=pypi&label=Latest%20PyPI%20Version&color=blue)](https://pypi.org/project/IMDLBenCo/)
![license](https://img.shields.io/github/license/scu-zjz/IMDLBenCo?logo=license)

<!----
[![Ask Me Anything!](https://img.shields.io/badge/Official%20-Yes-1abc9c.svg)](https://GitHub.com/scu-zjz/) 
---->

## News

- [2024/12/10] [Mesorch](https://arxiv.org/pdf/2412.13753) [![Stars](https://img.shields.io/github/stars/scu-zjz/Mesorch?style=social)](https://github.com/scu-zjz/Mesorch), our new IML backbone model, which adopts a parallel CNN+Transformer structure to simultaneously deal with image semantics and non-semantics, is accepted by AAAI 25!!!🎉🎉🎉
   
- [2024/12/10] [Sparse-ViT](https://arxiv.org/pdf/2412.14598) [![Stars](https://img.shields.io/github/stars/scu-zjz/SparseViT?style=social)](https://github.com/scu-zjz/SparseViT), the very first solution of constructing the non-semantic feature extractor through a self-supervised manner in IML is proposed by us and accepted by AAAI 25!!!🎉🎉🎉
- [2024/09/26] This paper, IMDL-BenCo, has been accepted as **Spotlight** to NeurIPS 2024 Track Datasets and Benchmarks!!! 🎉🎉🎉

> [!WARNING]
> **Existing bug**:
>    1. The `Image-Acc` evaluator may not return an accurate value under the current version during multi-GPU inference. Please be careful with it! We are locating the exact issue. Details found in issue: https://github.com/scu-zjz/IMDLBenCo/issues/42. Thanks [@iamwangyabin](https://github.com/iamwangyabin) for reporting the issue.

## Overview
☑️**Welcome to IMDL-BenCo, the first comprehensive IMDL benchmark and modular codebase.**    
- This codebase is under long-term maintenance and updating. New features, extra baseline/sota models, and bug fixes will be continuously involved. You can find the corresponding plan here shortly.
- This repo decomposes the IMDL framework into **standardized, reusable components and revises the model construction pipeline**, improving coding efficiency and customization flexibility.
- This repo **fully implements or incorporates training code for state-of-the-art models** to establish a comprehensive IMDL benchmark.
- Cite and star if you feel helpful. This will encourage us a lot 🥰.   

![](./images/IMDLBenCo_overview.png)

## Features under developing
This repository has completed training, testing, robustness testing, Grad-CAM, and other functionalities for mainstream models.

However, more features are currently being tested for improved user experience. Updates will be rolled out frequently. Stay tuned!

- [x] Install and download via PyPI
   - [x] You can experience on test PyPI now! 
- [x] Based on command line invocation, similar to `conda` in Anaconda.
   - [x] Dynamically create all training scripts to support personalized modifications.

- [ ] Information library, downloading, and re-management of IMDL datasets.
- [x] Support for Weight & Bias visualization.

## Quick Start
IMDL-BenCo is a Python library managed on PYPI now, it's easy to install by following the command:
```bash
pip install imdlbenco
```

For further guidance, please **click the buttons below** for official documentation:

[![Documents](https://img.shields.io/badge/Documents-Go_to_Main_Page-brightgreen?logo=read-the-docs)](https://scu-zjz.github.io/IMDLBenCo-doc/) [![Documents](https://img.shields.io/badge/中文文档-前往主页-brightgreen?logo=read-the-docs)](https://scu-zjz.github.io/IMDLBenCo-doc/zh/)

[![Documents](https://img.shields.io/badge/Documents-Go_to_Quick_start-brightgreen?logo=read-the-docs)](https://scu-zjz.github.io/IMDLBenCo-doc/guide/quickstart/install.html) [![Documents](https://img.shields.io/badge/中文文档-前往新手教程-brightgreen?logo=read-the-docs)](https://scu-zjz.github.io/IMDLBenCo-doc/zh/guide/quickstart/install.html)


We will keep updating the document with tricks and user cases. Please stay tuned!

We also welcome contributors to translate it into other languages.

## About
☑️**About the Developers:**  
- IMDL-BenCo's project leader/supervisor is Associate Professor 🏀[_Jizhe Zhou_ (周吉喆)](https://knightzjz.github.io/), Sichuan University🇨🇳.  
- IMDL-BenCo's codebase designer and coding leader is the Research Assitant [_Xiaochen Ma_ (马晓晨)](https://me.xiaochen.world/), Sichuan University🇨🇳.  
- IMDL-BenCo is jointly sponsored and advised by Prof. _Jiancheng LV_ (吕建成), Sichuan University 🐼, and Prof. _Chi-Man PUN_ (潘治文), University of Macau 🇲🇴, through the [Research Center of Machine Learning and Industry Intelligence, China MOE](https://center.dicalab.cn/) platform.  

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
## Statistics and Star History
<div align="center">

<a href="https://star-history.com/#scu-zjz/IMDLBenCo&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=scu-zjz/IMDLBenCo&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=scu-zjz/IMDLBenCo&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=scu-zjz/IMDLBenCo&type=Date" />
 </picture>
</a>

<a href="https://info.flagcounter.com/H5vw"><img src="https://s11.flagcounter.com/count2/H5vw/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_12/viewers_0/labels_0/pageviews_1/flags_0/percent_0/" alt="Flag Counter" border="0"></a></div>
