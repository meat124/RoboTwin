#!/bin/bash

policy_name=DiT_Graphormer_V2_MoGe 
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}
temporal_ensemble=${6}
# [TODO] add parameters here

SCRIPT_DIR=$(cd $(dirname $0) && pwd)
export PYTHONPATH=${SCRIPT_DIR}:${PYTHONPATH}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mtask name: ${task_name}\033[0m"
echo -e "\033[33mtask config: ${task_config}\033[0m"
echo -e "\033[33mcheckpoint setting: ${ckpt_setting}\033[0m"
echo -e "\033[33mseed: ${seed}\033[0m"
echo -e "\033[33mtemporal ensemble: ${temporal_ensemble}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name}  \
    --temporal_ensemble ${temporal_ensemble} \
    # [TODO] add parameters here
