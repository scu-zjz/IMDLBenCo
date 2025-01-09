base_dir="./robust_dir"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=1 \
./test_robust.py \
    --model ObjectFormer \
    --edge_mask_width 7 \
    --world_size 1 \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --checkpoint_path "/mnt/data0/username/workspace/IMDLBench/output_test/standard/output_dir_pretrain_224_1100_12_1e4_checkpoint-1052.pth" \
    --test_batch_size 2 \
    --image_size 224 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log