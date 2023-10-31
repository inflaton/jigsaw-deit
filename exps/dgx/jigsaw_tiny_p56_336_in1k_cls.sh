#!/bin/bash

export CUDA_VISIBLE_DEVICES="4"
export THREADS=1

python -m torch.distributed.launch \
    --nproc_per_node=$THREADS \
    --use_env \
    --master_port 40000 \
    main_jigsaw.py \
    --model jigsaw_tiny_patch56_336 \
    --input-size 336 \
    --permcls 50 \
    --batch-size 64 \
    --epochs 30 \
    --sched cosine \
    --unscale-lr \
    --lr 1e-3 \
    --min-lr 1e-8 \
    --mask-ratio 0.0 \
    --bce-loss \
    --data-path "/workspace/data/imagenet/ILSVRC/Data/CLS-LOC" \
    --data-set IMNET \
    --use-cls \
    --output_dir ./outputs/in1k_jigsaw_tiny_patch56_336_e10_c50ftc50_cls50


    # --finetune "./outputs/in1k_jigsaw_base_patch56_336_e10_c50ftc1000_cls*/checkpoint_29.pth" \
    # --finetune "./outputs/in1k_jigsaw_base_patch56_336_e10_c50ftc500_cls*/checkpoint_29.pth" \
    # --finetune "./outputs/in1k_jigsaw_base_patch56_336_e10_c50/checkpoint_9.pth" \
    # --output_dir ./outputs/debug
    # --finetune /workspace/study/jigsawvit/jigdeit/data/jigsaw_base_results/best_checkpoint.pth \