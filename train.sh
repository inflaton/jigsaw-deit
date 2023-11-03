#!/bin/sh
BASEDIR=$(dirname "$0")
cd $BASEDIR
echo Current Directory:
pwd

nvidia-smi

# use stable learning rate 1e-3
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export THREADS=4
# export WANDB_MODE=disabled
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
export THREADS=4
export CONFIG_ID="jigsaw_base_p56_336_f101_shuffle_in1ke80fte100_e50_1e-3_1024"

mkdir -p ./outputs/${CONFIG_ID}
mkdir -p ./logs/${CONFIG_ID}

python -m torch.distributed.launch \
    --nproc_per_node=$THREADS \
    --use_env \
    --master_port 40000 \
    main_jigsaw.py \
    --model jigsaw_base_patch56_336 \
    --input-size 336 \
    --batch-size 256 \
    --epochs 50 \
    --unscale-lr \
    --lr 1e-3 \
    --min-lr 1e-5 \
    --sched cosine \
    --mask-ratio 0.0 \
    --bce-loss \
    --data-path "./data/train/" \
    --data-set IMNET \
    --finetune "./data/checkpoints/best_checkpoint_e100.pth" \
    --use-cls \
    --output_dir ./outputs/${CONFIG_ID} \
    --log_dir ./logs/${CONFIG_ID}
