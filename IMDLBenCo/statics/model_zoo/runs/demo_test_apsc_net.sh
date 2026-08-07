base_dir="./eval_dir_apsc_net"
mkdir -p ${base_dir}

# Convert the official checkpoint first; see model_zoo/apsc_net/README.md:
# python -m IMDLBenCo.model_zoo.apsc_net.convert_checkpoint \
#   --input ./APSC-Net.pth --output-dir ./checkpoints/apsc_net/imdlbenco
CUDA_VISIBLE_DEVICES=0 \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
./test.py \
    --model APSCNet \
    --world_size 1 \
    --test_data_json "./test_datasets.json" \
    --checkpoint_path "./checkpoints/apsc_net/imdlbenco" \
    --test_batch_size 1 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --num_workers 4 \
2> ${base_dir}/error.log 1>${base_dir}/logs.log
