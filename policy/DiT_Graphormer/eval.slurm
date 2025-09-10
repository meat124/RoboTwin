#!/bin/bash

#SBATCH --job-name=policy_eval      # 잡 이름 (Job name)
#SBATCH --output=logs/%x_%j.out     # 표준 출력 로그 파일 (%x: job-name, %j: jobid)
#SBATCH --error=logs/%x_%j.err      # 에러 로그 파일
#SBATCH --partition=base_suma_rtx3090    # 사용할 파티션 이름
#SBATCH --gres=gpu:1                # GPU 1개 요청 (클러스터 설정에 맞게)
#SBATCH --time=12:00:00             # 실행 시간 제한 (HH:MM:SS)
# 그 외

module swap cuda/12.5 cuda/12.1
source /home/meat124/anaconda3/bin/activate RoboTwin_DiT_MoGe
conda activate RoboTwin_DiT_MoGe              # Conda 환경 활성화
echo "Conda environment activated."

# 실제 실행 명령
echo "Starting eval script in DiT_Graphormer directory."
# graphormer
srun bash eval.sh stack_blocks_three demo_clean /scratch2/meat124/dit_ws/src/RoboTwin/policy/DiT_Graphormer/bc_finetune/dit_robotwin_stack_blocks_three_demo_clean_seed0_monocular_diffusion_graphormer/wandb_dit_robotwin_stack_blocks_three_demo_clean_seed0_monocular_diffusion_graphormer_stack_blocks_three_resnet_gn_2025-09-02_20-50-25/dit_robotwin_stack_blocks_three_demo_clean_seed0_monocular_diffusion_graphormer.ckpt_best.ckpt 0 0 False

echo "Eval script completed."