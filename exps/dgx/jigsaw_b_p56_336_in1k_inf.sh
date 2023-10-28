#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
export THREADS=1

python -m torch.distributed.launch \
    --nproc_per_node=$THREADS \
    --use_env \
    --master_port 11000 \
    inference_perm.py \
    --eval \
    --model jigsaw_base_p56_336 \
    --input-size 336 \
    --batch-size 128 \
    --data-path /workspace/data/study/cspuzzle/336 \
    --data-set CSPUZZLE \
    --resume "./outputs/jigsaw_base_p56_336_in1k_e10/checkpoint_8.pth" \
    --output_dir ./preds/jigsaw_base_p56_336_in1k_e10

    # --finetune /workspace/study/jigsawvit/jigdeit/data/jigsaw_base_results/best_checkpoint.pth \