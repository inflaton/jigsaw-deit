#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ngpus=2
#PBS -l walltime=24:00:00
#PBS -P 21026844
#PBS -N jigsar_20ep
#PBS -m be
#PBS -M doem1997@gmail.com
#PBS -o ./exps/nscclog/s1_jigsar20e.log

# Commands start here
module load anaconda3/2022.10
conda activate study
cd ./study/puzzle/jigsaw-deit

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --use_env \
    --master_port 20000 \
    main_jigsaw.py \
    --rec \
    --model jigsar_base_p56_336 \
    --input-size 336 \
    --batch-size 1024 \
    --num_workers 32 \
    --epochs 20 \
    --sched cosine \
    --unscale-lr \
    --lr 1e-3 \
    --min-lr 1e-6 \
    --mask-ratio 0.0 \
    --data-path ./data/imagenet \
    --data-set IMNET \
    --lambda-rec 0.1 \
    --output_dir ./outputs/jigsar_b_p56_336_in1k_m0_20e

    # --resume "./outputs/jigsaw_base_p56_336_in1k/checkpoint_2.pth" \
    # --finetune /workspace/study/jigsawvit/jigdeit/data/jigsaw_base_results/best_checkpoint.pth \
