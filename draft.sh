# CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=1 \
./IMDLBench/training/train_backup.py \
    --world_size 1 \
    --batch_size 2 \
    --data_path /mnt/data0/public_datasets/IML/CASIA2.0_corrected \
    --epochs 200 \
    --lr 1e-4 \
    --if_resizing = Ture \
    --if_padding = False \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --edge_lambda 20 \
    --vit_pretrain_path /mnt/data0/xiaochen/workspace/IML-ViT/pretrained-weights/mae_pretrain_vit_base.pth \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --warmup_epochs 2 \
    --output_dir ./output_dir/ \
    --log_dir ./output_dir/  \
    --accum_iter 8 \
    --seed 42 \
    --test_period 4 \
2> train_error.log 1>train_logs.log