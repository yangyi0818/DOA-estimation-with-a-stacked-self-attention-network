##!/bin/bash

# Exit on error
set -e
set -o pipefail

export PYTHONPATH=/path/to/asteroid:/path/to/sms_wsj
python_path=

# General
stage=1
expdir=exp
id=$CUDA_VISIBLE_DEVICES

train_dir=
val_dir=
test_dir=
sample_rate=16000
. utils/parse_options.sh

# Training
batch_size=20
num_workers=16
optimizer=adam
lr=0.001
epochs=50

# Evaluation
eval_use_gpu=1

mkdir -p $expdir
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 1 ]]; then
  echo -e "Stage 1: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path -u train.py \
                --train_dirs $train_dir \
                --val_dirs $val_dir \
                --sample_rate $sample_rate \
                --lr $lr \
                --epochs $epochs \
                --batch_size $batch_size \
                --num_workers $num_workers \
                --exp_dir ${expdir}/ | tee logs/train.log
        cp logs/train.log $expdir/train.log
        echo -e "Stage 1 - training: Done."
fi

if [[ $stage -le 2 ]]; then
        echo -e "Stage 2 : Evaluation."
        echo -e "test set is $test_dir"
        CUDA_VISIBLE_DEVICES=$id $python_path -u eval.py \
                --test_dir $test_dir \
                --use_gpu $eval_use_gpu \
                --exp_dir ${expdir} | tee logs/eval.log
        cp logs/eval.log $expdir/eval.log
        echo -e "Stage 2 - evaluation: Done."
fi
