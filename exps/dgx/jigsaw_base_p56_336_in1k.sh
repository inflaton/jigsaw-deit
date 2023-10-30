#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export THREADS=4

python -m torch.distributed.launch \
    --nproc_per_node=$THREADS \
    --use_env \
    --master_port 20000 \
    main_jigsaw.py \
    --model jigsaw_base_patch56_336 \
    --input-size 336 \
    --permcls 1000 \
    --batch-size 512 \
    --epochs 100 \
    --sched cosine \
    --unscale-lr \
    --lr 1e-3 \
    --min-lr 1e-6 \
    --mask-ratio 0.0 \
    --bce-loss \
    --data-path "/workspace/data/imagenet/ILSVRC/Data/CLS-LOC" \
    --data-set IMNET \
    --finetune "./outputs/in1k_jigsaw_base_patch56_336_e10_c50/checkpoint_9.pth" \
    --output_dir ./outputs/in1k_jigsaw_base_patch56_336_e10_c1000

    # --finetune /workspace/study/jigsawvit/jigdeit/data/jigsaw_base_results/best_checkpoint.pth \