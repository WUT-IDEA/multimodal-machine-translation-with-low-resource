#!/bin/bash


SRC=en
TGT=uy # de

SAVE_DIR=results/$SRC-$TGT
ACMMT_ROOT=/data1/home/turghun/project/acmmt

export CUDA_VISIBLE_DEVICES=3
               
cp ${BASH_SOURCE[0]} ${SAVE_DIR}/train.sh

python train.py ${ACMMT_ROOT}/data_bin/$SRC-$TGT \
              --arch transformer \
              --share-decoder-input-output-embed --clip-norm 0 --optimizer adam --reset-optimizer --lr 0.001 \
              --source-lang $SRC --target-lang $TGT --max-tokens 1536 --no-progress-bar --log-interval 100 \
              --stop-min-lr 1e-09 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy \
              --label-smoothing 0.2 --lr-scheduler inverse_sqrt --max-update 4700 --warmup-updates 4000 \
              --warmup-init-lr 1e-07  --update-freq 4 --adam-betas 0.9,0.98 --keep-last-epochs 11 \
              --dropout 0.3 --tensorboard-logdir ${SAVE_DIR}/bl_log1 \
              --log-format simple --save-dir ${SAVE_DIR} --eval-bleu --maximize-best-checkpoint-metric


