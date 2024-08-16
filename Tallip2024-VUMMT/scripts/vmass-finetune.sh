#!/bin/bash

# mlm or vmlm or alter_vmlm

tgt="$1"
PRE="$2"

SRC=en
TGT=${tgt}


ROOT="/data1/home/turghun/project/VMLM"
DATA_PATH="${ROOT}/data/multi30k-${SRC}-${TGT}-half-la"
FEAT_PATH="/data1/home/turghun/project/images/features/faster/multi30k_oidv4_features/features"

DUMP_PATH=${ROOT}/models/vmass/${SRC}-${TGT}/latin
MLM_PATH="${ROOT}/pretrained/mass/${SRC}-${TGT}/latin/best-valid_${SRC}-${TGT}_${PRE}_bleu.pth"
EPOCH_SIZE=14500

export CUDA_VISIBLE_DEVICES=1

echo "-----------------model is initiaolized by ${MLM_PATH}---------------------"

python ../train.py  --beam_size 8 --exp_name ummt-fintune-${PRE}-${SRC}-${TGT}-s  --dump_path ${DUMP_PATH} \
    --data_path "${DATA_PATH}"  --reload_model $MLM_PATH,$MLM_PATH \
    --lgs "${SRC}-${TGT}"  --vae_steps "${SRC},${TGT}"  --vbt_steps "${SRC}-${TGT}-${SRC},${TGT}-${SRC}-${TGT}" --word_shuffle 3  \
    --word_dropout 0.1  --word_blank 0.1  --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false \
    --batch_size 64  --bptt 256 --emb_dim 512  --n_layers 6  --n_heads 8  \
    --dropout 0.2 --attention_dropout 0.1  --gelu_activation true \
    --inputs_concat false  --select_attn true  \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.998,lr=0.0001 --keep_best_checkpoints 11 \
    --epoch_size ${EPOCH_SIZE}  --eval_bleu true  --stopping_criterion "valid_${SRC}-${TGT}_mmt_bleu,10" \
    --validation_metrics "valid_${SRC}-${TGT}_mmt_bleu,valid_${TGT}-${SRC}_mmt_bleu" \
    --image_names $DATA_PATH  --region_feats_path $FEAT_PATH --num_of_regions 36   


