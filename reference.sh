model="ConvSegDctMoE"

epoch=200
accum_iter=4
dataset=balanced_dataset
dataset_dir=/mnt/data0/xuekang/workspace/convswin/${dataset}.json
batch=12
base_dir="${model}_${epoch}_accum_accit_${accum_iter}_${dataset}_${batch}"
output_dir="./output_dir_${base_dir}"
eval_dir="./eval_dir_output_dir_${base_dir}"
mkdir -p ${output_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
./IMDLBench/training/train_backup_model.py \
    --model ${model} \
    --world_size 1 \
    --batch_size ${batch} \
    --epochs ${epoch} \
    --lr 1e-4 \
    --data_path  ${dataset_dir} \
    --image_size 512 \
    --if_resizing \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --edge_lambda 20 \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --warmup_epochs 2 \
    --output_dir ${output_dir}/ \
    --log_dir ${output_dir}/ \
    --accum_iter ${accum_iter} \
    --seed 42 \
    --test_period 4 \
    --num_workers 24 \
2> ${output_dir}/error.log 1>${output_dir}/logs.log

mkdir -p ${eval_dir}
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
./IMDLBench/training/test_evaluators_model.py \
    --model ${model} \
    --world_size 1 \
    --test_data_json "./test_datasets.json" \
    --checkpoint_path "/mnt/data0/xuekang/workspace/convswin/output_dir_${base_dir}" \
    --test_batch_size ${batch} \
    --image_size 512 \
    --if_resizing \
    --output_dir ${eval_dir}/ \
    --log_dir ${eval_dir}/ \
2> ${eval_dir}/error.log 1>${eval_dir}/logs.log