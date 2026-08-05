#!/bin/bash
set -e
set -x

MODEL=ViT-B-32
PRETRAINED=laion2b_s34b_b79k
MAX_EPOCH=500
WARM_EPOCH=0
WORLD_SIZE=2

CKPT_DIR=ckpt/KP-Prediction-Heatmap-5+1Points-CPM-V2-FrontalProfile-LR-Aug-Scheduler-MSE
DATA_DIR=data/
LR=0.0001
WD=0.05
WD_END=""
NUM_CLASSES=101
BATCH_SIZE=32
EVAL_FREQ=1
SAVE_FREQ=10
NUM_WORKERS=8
#RESUME="--resume /home/ttdat/workspace/alignment/ckpt/KP-Prediction-Heatmap-5+1Points-CPM-V2-Frontal-LR-Scheduler-MSE/epoch-0150.pth"

torchrun --nnodes=1 --nproc_per_node=${WORLD_SIZE} \
        train_heatmap.py \
        --model ${MODEL} \
        --pretrained ${PRETRAINED} \
        --train_data ${DATA_DIR} \
        --val_data ${DATA_DIR} \
        --num_classes ${NUM_CLASSES} \
        --lr ${LR} \
        --wd ${WD} ${WD_END} \
        --batch_size ${BATCH_SIZE} \
        --eval_freq ${EVAL_FREQ} \
        --ckpt_dir ${CKPT_DIR} \
        --num_workers ${NUM_WORKERS} \
        --max_epoch ${MAX_EPOCH} --warmup_epochs ${WARM_EPOCH} \
        --save_freq ${SAVE_FREQ} ${RESUME}

