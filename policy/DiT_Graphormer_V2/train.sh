#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}

export AC_CHUNK=16
export BATCH_SIZE=150

# export EXP_NAME="dit_robotwin_${task_name}_${task_config}_seed${seed}_img_chunk_1_ac_chunk_${AC_CHUNK}_batch_${BATCH_SIZE}_diffusion_graphormer_v2"
# export WANDB_NAME="dit_robotwin_${task_name}_${task_config}_seed${seed}_img_chunk_1_ac_chunk_${AC_CHUNK}_batch_${BATCH_SIZE}_diffusion_graphormer_v2"
export EXP_NAME="test"
export WANDB_NAME="test"

export DATA_DIR="/scratch2/meat124/dit_ws/src/RoboTwin/data_dit/${task_name}/${task_config}"

export RESTORE_PATH="/scratch2/meat124/dit_ws/src/RoboTwin/policy/DiT_Graphormer_V2/visual_features/resnet18/IN_1M_resnet18.pth"

# export RESUME_PATH="$EXP_NAME.ckpt_100000.ckpt"
export RESUME_PATH=null


echo "============================================="
echo "Starting DiT_Graphormer_V2 Policy Training for RoboTwin"
echo "Experiment Name: $EXP_NAME"
echo "Dataset Path: $DATA_DIR/buf.pkl"
echo "============================================="

python finetune.py \
    exp_name=$EXP_NAME \
    agent=diffusion_graphormer_v2 \
    task=${task_name} \
    buffer_path="$DATA_DIR/buf.pkl" \
    resume_path=$RESUME_PATH \
    agent/features=resnet_gn \
    agent.features.restore_path=$RESTORE_PATH \
    trainer=bc_cos_sched \
    ac_chunk=$AC_CHUNK \
    batch_size=$BATCH_SIZE \
    wandb.name=$WANDB_NAME \
    wandb.mode=disabled
echo "============================================="
echo "Training Finished."
echo "============================================="