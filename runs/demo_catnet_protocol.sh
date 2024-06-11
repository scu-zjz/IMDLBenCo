base_dir="./output_dir_balance"
mkdir -p ${base_dir}

# CUDA_VISIBLE_DEVICES=3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
./IMDLBench/training/train_backup.py \
    --world_size 1 \
    --batch_size 2 \
    --epochs 200 \
    --lr 1e-4 \
    --data_path  /mnt/data0/xiaochen/workspace/IMDLBench/balanced_dataset.json \
    --image_size 1024 \
    --if_resizing \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --edge_lambda 20 \
    --vit_pretrain_path /mnt/data0/xiaochen/workspace/IML-ViT/pretrained-weights/mae_pretrain_vit_base.pth \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --warmup_epochs 2 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 8 \
    --seed 42 \
    --test_period 4 \
2> ${base_dir}/error.log 1>${base_dir}/logs.log