#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export THREADS=8
export CONFIG_ID="jigsaw_small_p56_336_in1k_c1000frcl50_nh_bs10241e-8e100"

python -m torch.distributed.launch \
    --nproc_per_node=$THREADS \
    --use_env \
    --master_port 40000 \
    main_jigsaw.py \
    --model jigsaw_small_patch56_336 \
    --input-size 336 \
    --permcls 1000 \
    --batch-size 128 \
    --epochs 100 \
    --sched cosine \
    --unscale-lr \
    --lr 1e-3 \
    --min-lr 1e-8 \
    --mask-ratio 0.0 \
    --bce-loss \
    --data-path "/workspace/data/imagenet/ILSVRC/Data/CLS-LOC" \
    --finetune "./outputs/in1k_jigsaw_small_patch56_336_e30_c1000/best_checkpoint.pth" \
    --data-set IMNET \
    --use-cls \
    --freeze \
    --output_dir ./outputs/${CONFIG_ID} \
    --log_dir ./logs/${CONFIG_ID}

    # --finetune "./outputs/in1k_jigsaw_base_patch56_336_e10_c50ftc500_cls*/checkpoint_29.pth" \
    # --finetune "./outputs/in1k_jigsaw_base_patch56_336_e10_c50/checkpoint_9.pth" \
    # --output_dir ./outputs/debug
    # --finetune /workspace/study/jigsawvit/jigdeit/data/jigsaw_base_results/best_checkpoint.pth \