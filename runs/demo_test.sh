base_dir="./eval_dir"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=3 \
./IMDLBench/training/test_evaluators.py \
    --world_size 1 \
    --test_data_json "./test_datasets.json" \
    --checkpoint_path "/mnt/data0/xiaochen/workspace/IMDLBench/output_dir" \
    --test_batch_size 3 \
    --image_size 1024 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log