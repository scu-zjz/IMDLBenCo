Please refer to [CAT-Net official repository](https://github.com/mjkwon2021/CAT-Net) to download pre-trained weights.

It should include two files:
- `hrnetv2_w48_imagenet_pretrained.pth`
- `DCT_djpeg.pth.tar`

And you need to revise the path of following key-value pairs in `CAT_full.yaml` under the `config` dir after `benco init model_zoo`:
```yaml
  PRETRAINED_RGB: '/mnt/data0/bingkui/Cat-Net/hrnetv2_w48_imagenet_pretrained.pth'
  PRETRAINED_DCT: '/mnt/data0/bingkui/Cat-Net/DCT_djpeg.pth.tar'
```