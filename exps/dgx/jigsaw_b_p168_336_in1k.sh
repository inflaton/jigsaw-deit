#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export THREADS=4

python -m torch.distributed.launch \
    --nproc_per_node=$THREADS \
    --use_env \
    --master_port 20000 \
    main_jigsaw.py \
    --model jigsaw_base_patch168_336 \
    --input-size 336 \
    --batch-size 512 \
    --epochs 10 \
    --sched cosine \
    --unscale-lr \
    --lr 1e-3 \
    --min-lr 1e-6 \
    --mask-ratio 0.0 \
    --bce-loss \
    --data-path "/workspace/data/imagenet/ILSVRC/Data/CLS-LOC" \
    --data-set IMNET \
    --output_dir ./outputs/in1k_jigsaw_base_patch168_336_e10

    # --resume "./outputs/jigsaw_base_patch168_336_in1k/checkpoint_2.pth" \
    # --finetune /workspace/study/jigsawvit/jigdeit/data/jigsaw_base_results/best_checkpoint.pth \