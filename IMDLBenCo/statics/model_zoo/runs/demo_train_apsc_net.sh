base_dir="./output_dir_apsc_net"
mkdir -p ${base_dir}

# This is an experimental fine-tuning launcher for the released full
# checkpoint. It is not a reproduction of the paper's 160k-iteration,
# eight-GPU, multi-source training protocol. The released inference head has
# no image-level classification loss in the public training configuration, so
# do not use this script as evidence of from-scratch training reproduction.
CUDA_VISIBLE_DEVICES=0 \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
./train.py \
    --model APSCNet \
    --pretrained "./checkpoints/apsc_net/APSC-Net.pth" \
    --world_size 1 \
    --batch_size 2 \
    --accum_iter 1 \
    --data_path "./balanced_dataset.json" \
    --epochs 200 \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --weight_decay 0.05 \
    --warmup_epochs 2 \
    --image_size 512 \
    --if_resizing \
    --test_data_path "./data/CASIA1" \
    --test_period 2 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --seed 42 \
    --num_workers 4 \
2> ${base_dir}/error.log 1>${base_dir}/logs.log
