#!/bin/bash

export CUDA_VISIBLE_DEVICES="4,5,6,7"
export THREADS=4

python -m torch.distributed.launch \
    --nproc_per_node=$THREADS \
    --use_env \
    main_jigsaw.py \
    --model jigsaw_base_patch16_384 \
    --input-size 384 \
    --batch-size 56 \
    --epochs 300 \
    --sched cosine \
    --lr 1e-3 \
    --warmup-lr 1e-3 \
    --min-lr 1e-6 \
    --warmup-epochs 1 \
    --mask-ratio 0.0 \
    --lambda-jigsaw 0.1 \
    --data-path /workspace/data/MSCOCO \
    --data-set COCO \
    --output_dir ./outputs/jigsaw_base_patch16_384
    # --finetune /workspace/study/jigsawvit/jigdeit/data/jigsaw_base_results/best_checkpoint.pth \