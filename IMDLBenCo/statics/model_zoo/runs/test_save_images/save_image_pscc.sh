base_dir="./save_img/pscc"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=1 \
./save_images.py \
    --model PSCC_Net \
    --edge_mask_width 7 \
    --world_size 1 \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --pretrain_path "/mnt/data0/IMDLBenCo-main/new_workspace/weights/PSCC/hrnet_w18_small_v2.pth" \
    --checkpoint_path "/mnt/data0/public_datasets/IML/IMDLBenCo_ckpt/pscc_catnet.pth" \
    --test_batch_size 2 \
    --image_size 256 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log