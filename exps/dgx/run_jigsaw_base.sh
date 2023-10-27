#!/bin/bash

export CUDA_VISIBLE_DEVICES="4,5,6,7"
export THREADS=4

python -m torch.distributed.launch \
    --nproc_per_node=$THREADS \
    --use_env \
    main_jigsaw.py \
    --model jigsaw_base_patch16_384 \
    --batch-size 128 \
    --epochs 200 \
    --sched cosine \
    --lr 1e-3 \
    --warmup-lr 1e-6 \
    --min-lr 1e-6 \
    --warmup-epochs 1 \
    --mask-ratio 0.0 \
    --lambda-jigsaw 0.1 \
    --data-path /workspace/study/jigsawvit/jigdeit/data/cifar10 \
    --data-set IMNET \
    # --finetune /workspace/study/jigsawvit/jigdeit/data/jigsaw_base_results/best_checkpoint.pth \
    --output_dir ./outputs/jigsaw_base_patch16_384