base_dir="./robust_dir"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
./test_robust.py \
    --model IML_ViT \
    --edge_mask_width 7 \
    --world_size 1 \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --checkpoint_path "/mnt/data0/xiaochen/workspace/IML-VIT-shuffle/rebuttal_trufor/checkpoint-188.pth" \
    --test_batch_size 2 \
    --image_size 1024 \
    --if_padding \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log