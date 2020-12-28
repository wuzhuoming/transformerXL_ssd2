#!/bin/bash

URL=http://curtis.ml.cmu.edu/datasets/pretrained_xl

DATA_ROOT=./

function download () {
  fileurl=${1}
  filename=${fileurl##*/}
  if [ ! -f ${filename} ]; then
    echo ">>> Download '${filename}' from '${fileurl}'."
    wget --quiet ${fileurl}
  else
    echo "*** File '${filename}' exists. Skip."
  fi
}

cd $DATA_ROOT
mkdir -p pretrained_xl && cd pretrained_xl

# enwik8
mkdir -p tf_enwik8 && cd tf_enwik8

mkdir -p data && cd data
download ${URL}/tf_enwiki8/data/cache.pkl
download ${URL}/tf_enwiki8/data/corpus-info.json
cd ..
