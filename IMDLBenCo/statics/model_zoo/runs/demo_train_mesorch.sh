base_dir="./output_dir_mesorch"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
./train.py \
    --model Mesorch \
    --conv_pretrain True \
    --seg_pretrain_path "/mnt/data0/xuekang/workspace/segformer/mit_b3.pth" \
    --world_size 4 \
    --find_unused_parameters \
    --batch_size 12 \
    --data_path /mnt/data0/xuekang/workspace/Mesorch/balanced_dataset.json \
    --epochs 150 \
    --lr 1e-4 \
    --image_size 512 \
    --if_resizing \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --warmup_epochs 2 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 2 \
    --seed 42 \
    --test_period 2 \
    --num_workers 12 \
2> ${base_dir}/error.log 1>${base_dir}/logs.log