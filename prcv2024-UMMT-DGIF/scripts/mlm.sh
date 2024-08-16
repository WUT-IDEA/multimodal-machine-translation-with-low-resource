#!/bin/bash



tgt="$1"


SRC=en
TGT=${tgt}

ROOT="/data1/home/turghun/project/low"
DATA_PATH="/data1/home/turghun/project/VMLM/data/multi30k/mono/${SRC}-${TGT}"
DUMP_PATH=${ROOT}/models/${SRC}-${TGT}

EPOCH_SIZE=14500


export CUDA_VISIBLE_DEVICES=3

python ../train.py --exp_name mlm-multi30k --dump_path $DUMP_PATH --data_path $DATA_PATH \
  --lgs "${SRC}-${TGT}" --clm_steps '' --mlm_steps "${SRC},${TGT}" \
  --emb_dim 512 --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' --gelu_activation true --batch_size 64 --bptt 256 \
  --optimizer "adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,warmup_updates=4000" \
  --tokens_per_batch 2000 --epoch_size ${EPOCH_SIZE} --max_epoch 100000 \
  --validation_metrics '_valid_mlm_ppl' --stopping_criterion '_valid_mlm_ppl,25' \
  --fp16 false --keep_best_checkpoints 11 --save_periodic 5
  
