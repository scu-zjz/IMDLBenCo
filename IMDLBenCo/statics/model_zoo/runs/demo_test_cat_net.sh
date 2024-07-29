base_dir="./eval_dir"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=3 \
./test.py \
    --model Cat_Net \
    --world_size 1 \
    --test_data_json "./test_datasets.json" \
    --checkpoint_path "/home/bingkui/IMDLBenCo/output_dir_balance" \
    --test_batch_size 3 \
    --if_resizing \
    --image_size 512 \
    --cfg_file "/home/bingkui/workspace/IMDLBenCo/configs/CAT_full.yaml" \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log