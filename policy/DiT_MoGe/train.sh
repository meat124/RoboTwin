#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}

export AC_CHUNK=16
export BATCH_SIZE=64

export EXP_NAME="dit_robotwin_${task_name}_${task_config}_seed${seed}_monocular_diffusion_moge_dformerv2"
export WANDB_NAME="dit_robotwin_${task_name}_${task_config}_seed${seed}_monocular_diffusion_moge_dformerv2"
# export EXP_NAME="test"
# export WANDB_NAME="test"

export DATA_DIR="/scratch2/meat124/dit_ws/src/RoboTwin/data_dit/${task_name}/${task_config}"

export RESTORE_PATH="/scratch2/meat124/dit_ws/src/RoboTwin/policy/DiT_MoGe/visual_features/dformerv2/DFormerv2_Small_NYU.pth"

# export RESUME_PATH="/scratch2/meat124/dit_ws/src/RoboTwin/policy/DiT_MoGe/bc_finetune/dit_robotwin_stack_blocks_three_demo_clean_seed0_monocular_diffusion_moge_dformerv2/wandb_dit_robotwin_stack_blocks_three_demo_clean_seed0_monocular_diffusion_moge_dformerv2_stack_blocks_three_dformerv2_2025-09-16_14-06-45/dit_robotwin_stack_blocks_three_demo_clean_seed0_monocular_diffusion_moge_dformerv2.ckpt_best.ckpt"
export RESUME_PATH=null


echo "============================================="
echo "Starting DiT_MoGe Policy Training for RoboTwin"
echo "Experiment Name: $EXP_NAME"
echo "Dataset Path: $DATA_DIR/buf_moge_enc_normalized.pkl"
echo "============================================="

python finetune.py \
    exp_name=$EXP_NAME \
    agent=diffusion_moge \
    task=${task_name} \
    buffer_path="$DATA_DIR/buf_moge_enc_normalized.pkl" \
    resume_path=$RESUME_PATH \
    agent/features=dformerv2 \
    agent.features.restore_path=$RESTORE_PATH \
    trainer=bc_cos_sched \
    ac_chunk=$AC_CHUNK \
    batch_size=$BATCH_SIZE \
    wandb.name=$WANDB_NAME \
    wandb.mode=online
echo "============================================="
echo "Training Finished."
echo "============================================="