# APSC-Net

This directory provides an IMDLBenCo adapter for APSC-Net from:

> Qu et al., *Towards Modern Image Manipulation Localization: A Large-Scale
> Dataset and Novel Methods*, CVPR 2024.

- Paper: <https://openaccess.thecvf.com/content/CVPR2024/html/Qu_Towards_Modern_Image_Manipulation_Localization_A_Large-Scale_Dataset_and_Novel_CVPR_2024_paper.html>
- Official code: <https://github.com/qcf-568/MIML>
- Official checkpoint: <https://drive.google.com/file/d/1fTFUnn1mCO9w-YG3wa9Xqqkdn2PsSwmZ/view>
- Upstream license notice: [UPSTREAM_NOTICE.md](./UPSTREAM_NOTICE.md)

The APSC-Net adapter directory is distributed under CC BY-NC 4.0 and is not
covered by IMDLBenCo's repository-level CC BY 4.0 license. See
[LICENSE](./LICENSE) and [UPSTREAM_NOTICE.md](./UPSTREAM_NOTICE.md).

## Integration scope

The official implementation depends on PyTorch 1.13, MMCV 1.6, and MMSEG.
This adapter implements the inference-relevant ConvNeXt backbone, APSC decoder,
three refinement passes, classification branch, and OHEM segmentation loss
directly with PyTorch. It does not require MMCV or MMSEG at runtime.

The model is registered as `APSCNet` and returns the standard IMDLBenCo keys:

- `backward_loss`
- `pred_mask`, shaped `[B, 1, H, W]`
- `pred_label`, shaped `[B]`
- `visual_loss`
- `visual_image`

IMDLBenCo already applies ImageNet normalization. This is numerically
equivalent to the pixel-space mean and standard deviation in the official
APSC-Net test configuration, so the adapter does not normalize the input a
second time.

## Official checkpoint

The checkpoint used for compatibility testing has:

- File name: `APSC-Net.pth`
- Size: `572,856,509` bytes
- SHA-256: `1515ef3c462a7e2908489f5627610dabc14e816623638167a0fc000fa415c62b`
- State entries: `578`
- Strict loading result: zero missing and zero unexpected keys

The checkpoint is not stored in this repository. Download it from the official
project and verify the hash before testing.

For direct Python inference, the official file can be loaded by the model:

```python
from IMDLBenCo.model_zoo import APSCNet

model = APSCNet(
    pretrained="./APSC-Net.pth",
    strict_load=True,
)
```

For the generic IMDLBenCo `test.py` runner, wrap the state dictionary in the
framework checkpoint format:

```bash
python -m IMDLBenCo.model_zoo.apsc_net.convert_checkpoint \
  --input ./APSC-Net.pth \
  --output-dir ./checkpoints/apsc_net/imdlbenco
```

Then set `--checkpoint_path ./checkpoints/apsc_net/imdlbenco` in
`runs/demo_test_apsc_net.sh`.

## Input and evaluation protocol

- Input policy: resize to `512 x 512`
- Numerical precision used in the audited run: FP32
- Batch size used in the audited run: 1
- Generic IMDLBenCo mask threshold: 0.5

The official APSC-Net inference code can instead derive a threshold in the
range 0.3 to 0.7 from the classification score. The generic IMDLBenCo runner
currently receives the probability mask and applies its normal metric
protocol. Results from the two threshold policies must therefore be labelled
separately.

An audited fixed-threshold run on CASIA v1 used 1,720 images, with pixel-level
metrics averaged over the 920 manipulated images:

| Metric | Result |
|---|---:|
| Pixel-F1 | 0.8412 |
| Pixel-AUC | 0.9829 |
| Image-F1 | 0.8929 |
| Image-AUC | 0.9610 |

The paper reports a CASIA v1 Pixel-F1 of 0.848. The audited result differs by
-0.0068. This comparison is a checkpoint-inference reproduction, not a
from-scratch training reproduction.

## Training limitation

The paper uses a multi-source dataset, AdamW, a polynomial schedule with 1,500
warm-up iterations, and 160,000 iterations on eight GPUs. The supplied
`demo_train_apsc_net.sh` is only an experimental, epoch-based fine-tuning
launcher for the released full checkpoint. It does not reproduce the paper's
complete training protocol, and it should not be described as such.

## Citation

```bibtex
@inproceedings{qu2024towards,
  title={Towards Modern Image Manipulation Localization: A Large-Scale Dataset and Novel Methods},
  author={Qu, Chenfan and Zhong, Yiwu and Liu, Chongyu and Xu, Guitao and Peng, Dezhi and Guo, Fengjun and Jin, Lianwen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10781--10790},
  year={2024}
}
```
