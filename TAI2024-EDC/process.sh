#!/bin/bash

src='uy'
tgt='en'

RAW_TEXT=data-raw
OUTPUT=data_bin/$src-$tgt

rm -rf data_bin

fairseq-preprocess \
  --source-lang $src \
  --target-lang $tgt \
  --trainpref ${RAW_TEXT}/train \
  --validpref ${RAW_TEXT}/valid \
  --testpref $RAW_TEXT/test \
  --destdir $OUTPUT \

