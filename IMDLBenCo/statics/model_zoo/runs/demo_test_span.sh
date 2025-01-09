base_dir="./eval_dir"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=3 \
./test.py \
    --model SPAN \
    --edge_mask_width 7 \
    --world_size 1 \
    --test_data_json "./test_datasets.json" \
    --checkpoint_path "/home/zeyu/workspace/IMDLBenCo/output_dir" \
    --test_batch_size 3 \
    --image_size 224 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --weight_path '/home/zeyu/workspace/IMDLBenCo/IMDLBenCo/model_zoo/span/IMTFEv4.pt' \
2> ${base_dir}/error.log 1>${base_dir}/logs.log