#!/bin/bash

#SBATCH --job-name=policy_train         # 잡 이름 (Job name)
#SBATCH --output=logs/%x_%j.out     # 표준 출력 로그 파일 (%x: job-name, %j: jobid)
#SBATCH --error=logs/%x_%j.err      # 에러 로그 파일
#SBATCH --partition=suma_a6000        # 사용할 파티션 이름
#SBATCH --gres=gpu:1                # GPU 1개 요청 (클러스터 설정에 맞게)
#SBATCH --time=24:00:00             # 실행 시간 제한 (HH:MM:SS)
# 그 외

module swap cuda/12.5 cuda/12.1
source /home/meat124/anaconda3/bin/activate RoboTwin_DiT
conda activate RoboTwin_DiT              # Conda 환경 활성화
echo "Conda environment activated."

# 실제 실행 명령
echo "Starting eval script in DiT directory."
# bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id}
# renewed
# srun bash eval.sh stack_blocks_three demo_clean /scratch2/meat124/dit_ws/src/RoboTwin/policy/DiT/bc_finetune/dit_robotwin_stack_blocks_three_demo_clean_seed0_img_chunk_1_ac_chunk_16_batch_150_renewed/wandb_dit_robotwin_stack_blocks_three_demo_clean_seed0_img_chunk_1_ac_chunk_16_batch_150_renewed_stack_blocks_three_resnet_gn_2025-08-17_00-41-49/dit_robotwin_stack_blocks_three_demo_clean_seed0_img_chunk_1_ac_chunk_16_batch_150_renewed.ckpt 0 0
# acdim_seq
# srun bash eval.sh stack_blocks_three demo_clean /scratch2/meat124/dit_ws/src/RoboTwin/policy/DiT/bc_finetune/dit_robotwin_stack_blocks_three_demo_clean_seed0_img_chunk_1_ac_chunk_16_batch_150_acdim_seq/wandb_dit_robotwin_stack_blocks_three_demo_clean_seed0_img_chunk_1_ac_chunk_16_batch_150_acdim_seq_stack_blocks_three_resnet_gn_2025-08-17_15-31-28/dit_robotwin_stack_blocks_three_demo_clean_seed0_img_chunk_1_ac_chunk_16_batch_150_acdim_seq.ckpt 0 0
# graphormer
srun bash eval.sh stack_blocks_three demo_clean /scratch2/meat124/dit_ws/src/RoboTwin/policy/DiT/bc_finetune/dit_robotwin_stack_blocks_three_demo_clean_seed0_img_chunk_1_ac_chunk_16_batch_150_graphormer/wandb_dit_robotwin_stack_blocks_three_demo_clean_seed0_img_chunk_1_ac_chunk_16_batch_150_graphormer_stack_blocks_three_resnet_gn_2025-08-17_15-34-19/dit_robotwin_stack_blocks_three_demo_clean_seed0_img_chunk_1_ac_chunk_16_batch_150_graphormer.ckpt 0 0
# graphormer_flat (140k iter), diffusion_graphormer.yaml 수정 필요
# srun bash eval.sh stack_blocks_three demo_clean /scratch2/meat124/dit_ws/src/RoboTwin/policy/DiT/bc_finetune/dit_robotwin_stack_blocks_three_demo_clean_seed0_img_chunk_1_ac_chunk_16_batch_150_graphormer_flat/wandb_dit_robotwin_stack_blocks_three_demo_clean_seed0_img_chunk_1_ac_chunk_16_batch_150_graphormer_flat_stack_blocks_three_resnet_gn_2025-08-19_00-07-35/dit_robotwin_stack_blocks_three_demo_clean_seed0_img_chunk_1_ac_chunk_16_batch_150_graphormer_flat.ckpt_140000.ckpt 0 0

echo "Eval script completed."