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

REGION_FEAT_PATH="/data1/home/turghun/project/images/features/faster/multi30k_oidv4_features/orign"
GRID_FEAT_PATH="/data1/home/turghun/project/images/features/resnet-ummt"
GLOBAL_FEAT_PATH="/data1/home/turghun/project/images/features/resnet50/global_feature/split"

ROOT="/data1/home/turghun/project/low" 
DATA_PATH="/data1/home/turghun/project/VMLM/data/multi30k-zh-uy-hole"

CHECK="${ROOT}/models/zh-uy/${CKPT}/check/best-valid_${SRC}-${TGT}_mt_bleu.pth"

BS=${BS:-8}
NAME="${CKPT/.pth/}_beam${BS}/"
DUMP_PATH="${ROOT}/translation/${SRC}-${TGT}/with-img"

REGION=true
GRID=false
GLOBAL=false

TEST=test
BCS=64

echo "===================== Model path:${CHECK} ======================= "

export CUDA_VISIBLE_DEVICES=0

  echo ""
  echo "=======================$TEST======================"
  echo ""

python ../train.py --beam_size $BS --exp_name $NAME --exp_id "${TEST}" --dump_path ${DUMP_PATH} \
  --reload_model "${CHECK},${CHECK}" --data_path "${DATA_PATH}" \
  --encoder_only false --lgs "${SRC}-${TGT}" --mmt_step "${SRC}-${TGT}" --emb_dim 512 --n_layers 6 \
  --n_heads 8 --dropout '0.1' --attention_dropout '0.1' --gelu_activation true \
  --batch_size ${BCS} --optimizer "adam,lr=0.0001" --inputs_concat true --select_attn false \
  --eval_bleu true --eval_only true --num_of_regions 36 \
  --image_names $DATA_PATH  --region_feats_path $REGION_FEAT_PATH --grid_feats_path $GRID_FEAT_PATH --image_names $DATA_PATH \
  --re_img_feats $REGION --gr_img_feats $GRID --gl_img_feats $GLOBAL --global_feats_path $GLOBAL_FEAT_PATH

for TEST_SET in test_2017_flickr test_2017_mscoco; do

  # Binarized folder structure for the codebase recognizes only one test set
  # Create separate subfolders for this to work.
  echo ""
  echo "=======================$TEST_SET======================"
  echo "${DATA_PATH}/${TEST_SET}"

  python ../train.py --beam_size $BS --exp_name $NAME --exp_id "${TEST_SET}" --dump_path ${DUMP_PATH} \
    --reload_model "${CHECK},${CHECK}" --data_path "${DATA_PATH}/${TEST_SET}" \
    --encoder_only false --lgs "${SRC}-${TGT}" --mmt_step "${SRC}-${TGT}" --emb_dim 512 \
    --n_layers 6 --n_heads 8 --dropout '0.1' --attention_dropout '0.1' --gelu_activation true \
    --batch_size ${BCS} --optimizer "adam,lr=0.0001" --inputs_concat true --select_attn false  \
    --eval_bleu true --eval_only true --num_of_regions 36   \
    --re_img_feats $REGION --gr_img_feats $GRID --gl_img_feats $GLOBAL --global_feats_path $GLOBAL_FEAT_PATH \
    --image_names $DATA_PATH  --region_feats_path $REGION_FEAT_PATH --grid_feats_path $GRID_FEAT_PATH --image_names "${DATA_PATH}/${TEST_SET}" $@ 
done
