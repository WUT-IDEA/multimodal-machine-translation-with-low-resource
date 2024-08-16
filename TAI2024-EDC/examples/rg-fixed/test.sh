#!/bin/bash

SRC=en
TGT=fr # de, fr

TEST=testmscoco #test2016, test2017, testmscoco

IMG_DATA_PREFIX=/data1/home/turghun/project/images/features
SAVE_DIR=results/$SRC-$TGT/mmt
ACMMT_ROOT=/data1/home/turghun/project/acmmt
ANNEAL_ROOT=$ACMMT_ROOT/examples/rg-fixed

MODEL_PATH=${SAVE_DIR}/model.pt

#cp ${BASH_SOURCE[0]} ${SAVE_DIR}/train.sh

export CUDA_VISIBLE_DEVICES=0



fairseq-generate ${ACMMT_ROOT}/data_bin/$SRC-$TGT/${TEST} --user-dir ${ANNEAL_ROOT} --batch-size 64 \
        --criterion rgfix_criterion --task rgfix_translation_task \
        --path ${MODEL_PATH} --remove-bpe --nbest 1 --num-workers 12 \
        --source-lang ${SRC} --target-lang ${TGT} --beam 8 --lenpen 0.3 \
        --img-grid-prefix ${IMG_DATA_PREFIX}/resnet101-dlmmt/${TEST}.npy \
        --img-region-prefix ${IMG_DATA_PREFIX}/faster-dlmmt/${TEST}.pkl \
        --results-path ${SAVE_DIR}/${TEST}







                              