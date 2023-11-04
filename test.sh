#!/bin/sh
BASEDIR=$(dirname "$0")
cd $BASEDIR
echo Current Directory:
pwd

nvidia-smi

BATCH=768

python test.py -b $BATCH 
# -t data/cs/train/49/

tail data/test.txt
