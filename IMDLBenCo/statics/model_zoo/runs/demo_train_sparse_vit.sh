base_dir="./output_dir"
mkdir -p ${base_dir}

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
train.py \
    --model SparseViT \
    --world_size 1 \
    --batch_size 16 \
    --data_path /mnt/data0/public_datasets/IML/CASIA2.0 \
    --epochs 200 \
    --lr 2e-4 \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --pretrained_path "/mnt/data0/xiaochen/workspace/IMDLBenCo_pure/test_sparse_vit/uniformer_image/uniformer_base_in1k.pth" \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --if_resizing \
    --find_unused_parameters \
    --warmup_epochs 4 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 16 \
    --seed 42 \
    --test_period 4 \
    --num_workers 8 \
    2> ${base_dir}/train_error.log 1>${base_dir}/train_log.log