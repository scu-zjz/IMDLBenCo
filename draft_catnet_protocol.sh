CUDA_VISIBLE_DEVICES=3,4,5 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=3 \
./IMDLBench/training/train_catnet_protocol_backup.py \
    --world_size 1 \
    --batch_size 2 \
    --epochs 200 \
    --lr 1e-4 \
    --image_size 1024 \
    --if_resizing \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --edge_lambda 20 \
    --vit_pretrain_path /mnt/data0/xiaochen/workspace/IML-ViT/pretrained-weights/mae_pretrain_vit_base.pth \
    --test_data_path "/home/psdz/Datasets/CASIA1.0" \
    --warmup_epochs 2 \
    --output_dir ./output_dir/ \
    --log_dir ./output_dir/  \
    --accum_iter 8 \
    --seed 42 \
    --test_period 4 \
2> train_error.log 1>train_logs.log