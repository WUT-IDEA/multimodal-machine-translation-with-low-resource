#!/bin/bash


tgt="$1"
SRC=zh
TGT=${tgt}

ROOT="/data1/home/turghun/project/VMLM"
DATA_PATH="${ROOT}/data/multi30k/mono/${SRC}-${TGT}-la"
FEAT_PATH="/data1/home/turghun/project/images/coco2014-multi30k/features/faster_oidv4_features"

DUMP_PATH=${ROOT}/models/${SRC}-${TGT}/latin

EPOCH_SIZE=14500


export CUDA_VISIBLE_DEVICES=3

python ../train.py --exp_name vmlm-multi30k --dump_path $DUMP_PATH --data_path $DATA_PATH \
  --lgs "${SRC}-${TGT}" --clm_steps '' --mlm_steps "${SRC},${TGT}" \
  --emb_dim 512 --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' --gelu_activation true --batch_size 64 --bptt 256 \
  --optimizer "adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,warmup_updates=4000" \
  --epoch_size ${EPOCH_SIZE} --max_epoch 100000 --fp16 false --keep_best_checkpoints 11 \
  --validation_metrics '_valid_vmlm_ppl' --stopping_criterion '_valid_vmlm_ppl,50' \
  --tokens_per_batch 2000 --image_names $DATA_PATH  --region_feats_path $FEAT_PATH --inputs_concat true \
  --num_of_regions 36 --only_vmlm true  --eval_vmlm true --region_mask_type mask 