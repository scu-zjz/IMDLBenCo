CUDA_VISIBLE_DEVICES=3,4,5 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=3 \
./IMDLBench/training/test_backup.py \
    --world_size 1 \
    --test_data_json "./test_datasets.json" \
    --checkpoint_path "/mnt/data0/xiaochen/workspace/IMDLBench/output_dir" \
    --test_batch_size 4 \
    --image_size 1024 \
    --if_resizing \
    --output_dir ./eval_dir/ \
    --log_dir ./eval_dir/  \
2> test_error.log 1>test_logs.log