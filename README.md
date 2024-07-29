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

- [x] Install and download via PyPI
   - [x] You can experience on test PyPI now! 
- [x] Based on command line invocation, similar to `conda` in Anaconda.
   - [x] Dynamically create all training scripts to support personalized modifications.

- [ ] Information library, downloading, and re-management of IMDL datasets.
- [x] Support for Weight & Bias visualization.

## Quick Start

Please check our official documentation, we provided an English version and a Chinese version:

[IMDL-BenCo: Main Page](https://scu-zjz.github.io/IMDLBenCo-doc/)

[IMDL-BenCo: Quick Start](https://scu-zjz.github.io/IMDLBenCo-doc/guide/quickstart/install.html)

We also welcome contributors to translate it into other languages.

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
