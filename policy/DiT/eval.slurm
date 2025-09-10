#!/bin/bash

#SBATCH --job-name=policy_eval      # 잡 이름 (Job name)
#SBATCH --output=logs/%x_%j.out     # 표준 출력 로그 파일 (%x: job-name, %j: jobid)
#SBATCH --error=logs/%x_%j.err      # 에러 로그 파일
#SBATCH --partition=base_suma_rtx3090    # 사용할 파티션 이름
#SBATCH --gres=gpu:1                # GPU 1개 요청 (클러스터 설정에 맞게)
#SBATCH --time=5:00:00             # 실행 시간 제한 (HH:MM:SS)
# 그 외

module swap cuda/12.5 cuda/12.1
source /home/meat124/anaconda3/bin/activate RoboTwin_DiT_MoGe
conda activate RoboTwin_DiT_MoGe              # Conda 환경 활성화
echo "Conda environment activated."

# 실제 실행 명령
echo "Starting eval script in DiT directory."
# diffusion
# srun bash eval.sh {task_name} {task_config} {ckpt_setting} {seed} {gpu_id} {temporal_ensemble}
srun bash eval.sh lift_pot demo_clean /scratch2/meat124/dit_ws/src/RoboTwin/policy/DiT/bc_finetune/dit_robotwin_lift_pot_demo_clean_seed0_monocular_diffusion/wandb_dit_robotwin_lift_pot_demo_clean_seed0_monocular_diffusion_lift_pot_resnet_gn_2025-09-09_18-58-59/dit_robotwin_lift_pot_demo_clean_seed0_monocular_diffusion.ckpt_best.ckpt 2 0 False

echo "Eval script completed."