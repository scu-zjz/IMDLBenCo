# Upstream notice for APSC-Net

This adapter is based on the public APSC-Net implementation from:

- Chenfan Qu, Yiwu Zhong, Chongyu Liu, Guitao Xu, Dezhi Peng, Fengjun Guo,
  and Lianwen Jin. *Towards Modern Image Manipulation Localization: A
  Large-Scale Dataset and Novel Methods*. CVPR 2024.
- Paper: <https://openaccess.thecvf.com/content/CVPR2024/html/Qu_Towards_Modern_Image_Manipulation_Localization_A_Large-Scale_Dataset_and_Novel_CVPR_2024_paper.html>
- Code: <https://github.com/qcf-568/MIML>

The upstream repository states that the project is licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This adapter
therefore retains the upstream attribution and non-commercial restriction.

## License scope

Files in this directory derived from the MIML implementation are distributed
under CC BY-NC 4.0. See [LICENSE](./LICENSE) for the complete license text.
They are not covered by IMDLBenCo's repository-level CC BY 4.0 license. Other
IMDLBenCo files remain under the repository-level license.

## Modifications

This IMDLBenCo adapter modifies the upstream implementation by:

- replacing the MMCV/MMSEG runtime components with plain PyTorch modules;
- adapting the forward interface to the IMDLBenCo output contract;
- adding checkpoint-prefix normalization and checkpoint conversion; and
- adding IMDLBenCo evaluation and experimental fine-tuning launchers.
