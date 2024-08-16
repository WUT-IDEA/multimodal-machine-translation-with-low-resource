#!/bin/bash

# Train UNMT with pretraining language model

tgt="$1"
PRE="$2"

SRC=en
TGT=${tgt}

ROOT="/data1/home/turghun/project"

MLM_PATH=/data1/home/turghun/project/low/models/en-fr/vmlm-multi30k-region/71wvdfet4k/best-valid_vmlm_ppl.pth
#"${ROOT}/pretrained/${SRC}-${TGT}/best-valid_${PRE}_ppl.pth"

DATA_PATH="${ROOT}/VMLM/data/multi30k/mono/${SRC}-${TGT}"
EPOCH_SIZE=14500
DUMP_PATH=${ROOT}/low/models/${SRC}-${TGT}



export CUDA_VISIBLE_DEVICES=2

echo "-----------------model is initiaolized by ${MLM_PATH}---------------------"

python ../train.py  --beam_size 8 --exp_name unmt-fintune-${PRE}_${SRC}-${TGT}  --dump_path ${DUMP_PATH} \
    --data_path "${DATA_PATH}" --reload_model "${MLM_PATH},${MLM_PATH}" \
    --lgs "${SRC}-${TGT}"  --ae_steps "${SRC},${TGT}"  --bt_steps "${SRC}-${TGT}-${SRC},${TGT}-${SRC}-${TGT}" --word_shuffle 3  \
    --word_dropout 0.1  --word_blank 0.1  --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false \
    --batch_size 64  --bptt 256 --emb_dim 512  --n_layers 6  --n_heads 8  \
    --dropout 0.1 --attention_dropout 0.1  --gelu_activation true \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.998,lr=0.0001 \
    --epoch_size ${EPOCH_SIZE}  --eval_bleu true  --stopping_criterion "valid_${SRC}-${TGT}_mt_bleu,10" \
    --validation_metrics "valid_${SRC}-${TGT}_mt_bleu,valid_${TGT}-${SRC}_mt_bleu" --eval_from 20