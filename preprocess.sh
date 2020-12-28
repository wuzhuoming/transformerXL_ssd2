#!/bin/bash

# Path
LOCAL_DIR=/root/cyliu/tftuner/selftf/tf_job/nlp/zmwu/transformer_xl/data/enwik8/
GSDATA=/root/cyliu/tftuner/selftf/tf_job/nlp/zmwu/transformer_xl/data/processed_enwik8

# Training
TGT_LEN=768
MEM_LEN=768
TRAIN_BSZ=32
VALID_BSZ=16

NUM_CORE=16

python data_utils.py \
        --data_dir=${LOCAL_DIR}/ \
        --dataset=enwik8 \
        --tgt_len=${TGT_LEN} \
        --per_host_train_bsz=${TRAIN_BSZ} \
        --per_host_valid_bsz=${VALID_BSZ} \
        --num_core_per_host=${NUM_CORE} \
        --num_passes=10 \
        --use_tpu=False \

SRC_PATTERN=train.bsz-${TRAIN_BSZ}.tlen-${TGT_LEN}.core-${NUM_CORE}*
gsutil cp ${LOCAL_DIR}/tfrecords/${SRC_PATTERN} ${GSDATA}/enwik8-tfrecords/

SRC_PATTERN=valid.bsz-${VALID_BSZ}.tlen-${TGT_LEN}.core-${NUM_CORE}*
gsutil cp ${LOCAL_DIR}/tfrecords/${SRC_PATTERN} ${GSDATA}/enwik8-tfrecords/
