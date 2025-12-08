#!/bin/bash

CHECKPOINT="/data/ucb_checkpoints/results2_freezed_0.7_lr/ratio_0.7/final_checkpoint/finetune-frozen-0.7_ratio_0.7.bin"
H5_DIR="/data/patches"
LR =1e-3

for ratio in  0.2 0.5
do
    OUTPUT_DIR="/data/ucb_checkpoints/test_from_0.7_to_${ratio}"
    LOG_FILE="training_freezed_${ratio}.log"
    RUN_NAME="finetune-frozen-${ratio}"

    echo ">>> Avvio training con frozen_keep_ratio = ${ratio}"
    echo "    Output dir: ${OUTPUT_DIR}"
    echo "    Log: ${LOG_FILE}"

    nohup python3 training_freeze.py \
        --stage1_checkpoint_path ${CHECKPOINT} \
        --frozen_keep_ratio ${ratio} \
        --h5_dir ${H5_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --run_name "${RUN_NAME}" \
        --num_train_epochs 50 \
        --learning_rate 0.001 \
        --train_batch_size 12 \
        --report_to wandb \
        > ${LOG_FILE} 2>&1

    echo ">>> **Completato** ratio = ${ratio}"
    echo "    Passo al prossimo..."
done

echo ">>> Tutti i training completati!"
