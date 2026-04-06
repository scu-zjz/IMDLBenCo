base_dir="./eval_robust_dir/MSCDI_Net"

mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
./test_robust.py \
    --model MSCDI_Net \
    --edge_mask_width 7 \
    --backbone_weights_path /mnt/public/public_models/segformer/imagenet1k_pretrained/mit_b4.pth \
    --world_size 4 \
    --test_batch_size 4 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir} \
    --log_dir ${base_dir} \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --checkpoint_path "./output_dir/MSCDI_Net/checkpoint-164.pth" \
2>${base_dir}/error.log 1>${base_dir}/logs.log
