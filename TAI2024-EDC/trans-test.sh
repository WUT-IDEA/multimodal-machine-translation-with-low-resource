#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

SRC=en
TGT=de # de, fr

SAVE_DIR=results/$SRC-$TGT
ACMMT_ROOT=/data1/home/turghun/project/acmmt
TEST=test2017  #test2016, test2017, testmscoco
fairseq-generate ${ACMMT_ROOT}/data_bin/$SRC-$TGT/${TEST} \
				--path ${SAVE_DIR}/model.pt \
				--source-lang ${SRC} --target-lang ${TGT} \
				--beam 4 --lenpen 0.3 --num-workers 12  --batch-size 1 \
				--remove-bpe --nbest 1 --results-path ${SAVE_DIR}/test/${TEST} \


