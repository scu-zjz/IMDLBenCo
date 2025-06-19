base_dir="./save_img_dir_catnet"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
./test_save_images.py \
    --model Cat_Net \
    --cfg_file /mnt/data0/xiaochen/workspace/IMDLBenCo_pure/test_cat/configs/CAT_full.yaml \
    --world_size 1 \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --checkpoint_path "/mnt/data0/public_datasets/IML/IMDLBenCo_ckpt/cat_net_cat_net.pth" \
    --test_batch_size 2 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log