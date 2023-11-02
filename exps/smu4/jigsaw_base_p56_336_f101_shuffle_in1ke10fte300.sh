#!/bin/bash

# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export THREADS=4
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export THREADS=4
export CONFIG_ID="jigsaw_base_p56_336_f101_shuffle_in1ke10fte300"
# export WANDB_MODE=disabled

python -m torch.distributed.launch \
    --nproc_per_node=$THREADS \
    --use_env \
    --master_port 40000 \
    main_jigsaw.py \
    --model jigsaw_base_patch56_336 \
    --input-size 336 \
    --batch-size 512 \
    --epochs 300 \
    --sched cosine \
    --unscale-lr \
    --lr 1e-3 \
    --min-lr 1e-8 \
    --mask-ratio 0.0 \
    --bce-loss \
    --data-path "./data/food101/" \
    --data-set IMNET \
    --finetune "./outputs/jigsaw_base_p56_336_f101_shuffle_e300/best_checkpoint.pth" \
    --output_dir ./outputs/${CONFIG_ID} \
    --log_dir ./logs/${CONFIG_ID}

    # --use-cls \
    # --finetune "./outputs/in1k_jigsaw_base_patch56_336_e10_c50ftc500_cls*/checkpoint_29.pth" \
    # --finetune "./outputs/in1k_jigsaw_base_patch56_336_e10_c50/checkpoint_9.pth" \
    # --output_dir ./outputs/debug
    # --finetune /workspace/study/jigsawvit/jigdeit/data/jigsaw_base_results/best_checkpoint.pth \