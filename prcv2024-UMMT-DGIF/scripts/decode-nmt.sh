#!/bin/bash

# Decodes all test sets for a given pretrained checkpoint
# Check the Checkpoint's folder to see the created folders that contain the
# hypotheses and refs.
CKPT="$1"
SRC="uy"
TGT="zh"
if [ -z $CKPT ]; then
  echo 'You need to provide a checkpoint .pth file for decoding.'
  exit 1
fi

shift 1

ROOT="/data1/home/turghun/project/low"
DATA_PATH="/data1/home/turghun/project/VMLM/data/multi30k-zh-uy-hole"
CHECK="${ROOT}/models/zh-uy/${CKPT}/check/best-valid_${SRC}-${TGT}_mmt_bleu.pth"

BS=${BS:-8}
NAME="${CKPT/.pth/}_beam${BS}/"
DUMP_PATH="${ROOT}/translation/${SRC}-${TGT}"
TEST=test

echo "===================== Model path:${CHECK} ======================= "

export CUDA_VISIBLE_DEVICES=2

  echo ""
  echo "=======================$TEST======================"
  echo ""

# Decode test_2016_flickr first
python ../train.py --beam_size $BS --exp_name $NAME --exp_id "${TEST}" --dump_path ${DUMP_PATH} \
  --reload_model "${CHECK},${CHECK}" --data_path "${DATA_PATH}" \
  --encoder_only false --lgs "${SRC}-${TGT}" --mt_step "${SRC}-${TGT}" --emb_dim 512 --n_layers 6 --n_heads 8 \
  --dropout '0.1' --attention_dropout '0.1' --gelu_activation true \
  --batch_size 64 --optimizer "adam,lr=0.0001" \
  --eval_bleu true --eval_only true $@

for TEST_SET in test_2017_flickr test_2017_mscoco; do
  # Binarized folder structure for the codebase recognizes only one test set
  # Create separate subfolders for this to work.
  echo ""
  echo "=======================$TEST_SET======================"
  echo "${DATA_PATH}/${TEST_SET}"

  python ../train.py --beam_size $BS --exp_name $NAME --exp_id "${TEST_SET}" --dump_path ${DUMP_PATH} \
    --reload_model "${CHECK},${CHECK}" --data_path "${DATA_PATH}/${TEST_SET}" \
    --encoder_only false --lgs "${SRC}-${TGT}" --mt_step "${SRC}-${TGT}" --emb_dim 512 --n_layers 6 --n_heads 8 \
    --dropout '0.1' --attention_dropout '0.1' --gelu_activation true \
    --batch_size 64 --optimizer "adam,lr=0.0001"  \
    --eval_bleu true --eval_only true $@
  
done
