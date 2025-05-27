- You should download MAE pre-trained weights from the official repo of the paper [Masked Autoencoders are scalable vision learners](https://arxiv.org/abs/2111.06377) before start training(fine-tuning) with our IML-ViT.
    - [official repo of MAE](https://github.com/facebookresearch/mae)
    - [download link of MAE pre-trained ViT-B](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)

- By default, our commend for training takes this dir as the place where pre-trained weights are. Therefore, you should place your downloaded weights in this directory, or re-direct to your custom dir of MAE weights. 