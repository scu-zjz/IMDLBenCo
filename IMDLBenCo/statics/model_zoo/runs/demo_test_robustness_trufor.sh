base_dir="./log/robust_trufor"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=1,2,3,4,5 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=5 \
./IMDLBenCo/training_scripts/test_robust.py \
    --model Trufor \
    --edge_mask_width 7 \
    --np_pretrain_weights "/mnt/data0/dubo/workspace/IMDLBenCo/IMDLBenCo/model_zoo/trufor/noiseprint.pth" \
    --mit_b2_pretrain_weights "/mnt/data0/dubo/workspace/IMDLBenCo/IMDLBenCo/model_zoo/trufor/mit_b2.pth" \
    --config_path "/mnt/data0/dubo/workspace/IMDLBenCo/configs/trufor.yaml" \
    --world_size 1 \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --checkpoint_path "/mnt/data0/dubo/workspace/IMDLBenCo/log/train_casiav2full_trufor/checkpoint.pth" \
    --test_batch_size 8 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log