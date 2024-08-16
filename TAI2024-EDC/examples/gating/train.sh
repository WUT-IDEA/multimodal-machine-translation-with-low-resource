#!/bin/bash

SRC=en
TGT=fr # de, fr

SAVE_DIR=results/nmt/$SRC-$TGT
ACMMT_ROOT=/data1/home/turghun/project/acmmt
GATING_ROOT=$ACMMT_ROOT/examples/gating


export CUDA_VISIBLE_DEVICES=1

 fairseq-train ${ACMMT_ROOT}/data_bin/$SRC-$TGT \
        --user-dir ${GATING_ROOT} --criterion my_label_smoothed_cross_entropy \
        --task attack_translation_task --arch my_arch --optimizer myadam --adam-betas 0.9,0.98 \
        --clip-norm 0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
        --reset-optimizer --lr 0.0005 --weight-decay 0.0001 --label-smoothing 0.2  --dropout 0.3 \
        --max-tokens 1536 --no-progress-bar --log-interval 100 --stop-min-lr 1e-09 --max-update 150000\
        --keep-last-epochs 12 --update-freq 4 --eval-bleu --maximize-best-checkpoint-metric \
        --save-dir ${SAVE_DIR} --share-decoder-input-output-embed --source-lang ${SRC} --target-lang ${TGT}\
        --mask-loss-weight 0.03 --tensorboard-logdir ${SAVE_DIR}/bl_log1 --log-format simple \

              
cp ${BASH_SOURCE[0]} ${SAVE_DIR}/train.sh



