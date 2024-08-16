#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
IMG_DATA_PREFIX=/data1/home/turghun/project/images/features
TEST=test2016  #test2016, test2017, testmscoco
python3 generate.py data-bin/${TEST} \
				--path results/en-de/mmt-img/model.pt \
				--source-lang en --target-lang de \
				--beam 5 --num-workers 12  --batch-size 64 \
				--results-path results/test/mmt-img/${TEST} \
				--img_grid_prefix ${IMG_DATA_PREFIX}/resnet101-dlmmt/${TEST}.npy \
				--img_region_prefix ${IMG_DATA_PREFIX}/faster-dlmmt/${TEST}.pkl \
				--remove-bpe --nbest 1
