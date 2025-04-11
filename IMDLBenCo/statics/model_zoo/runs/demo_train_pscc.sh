base_dir="./output_dir_pscc"
mkdir -p ${base_dir}
# Check point to hrnet_w18_small_v2.pth please reference here:
# https://github.com/proteus1991/PSCC-Net/blob/main/models/hrnet_w18_small_v2.pth

CUDA_VISIBLE_DEVICES=4 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=1 \
./train.py \
    --model PSCC_Net \
    --world_size 1 \
    --batch_size 16 \
    --data_path /mnt/data0/public_datasets/IML/CASIA2.0 \
    --pretrain_path "/mnt/data0/xiaochen/workspace/IMDLBenCo_pure/test_cat_net/hrnet_w18_small_v2.pth" \
    --epochs 150 \
    --lr 1e-4 \
    --image_size 256 \
    --if_resizing \
    --min_lr 0 \
    --weight_decay 0.05 \
    --edge_mask_width 7 \
    --if_predict_label \
    --if_not_amp \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --warmup_epochs 2 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 1 \
    --seed 42 \
    --test_period 4 \
2> ${base_dir}/error.log 1>${base_dir}/logs.log