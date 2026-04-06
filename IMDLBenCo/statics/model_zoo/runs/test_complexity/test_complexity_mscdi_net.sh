python ./test_complexity.py \
    --model MSCDI_Net \
    --test_batch_size 1 \
    --edge_mask_width 7 \
    --backbone_weights_path /mnt/public/public_models/segformer/imagenet1k_pretrained/mit_b4.pth \
    --image_size 512 \
    --if_resizing
