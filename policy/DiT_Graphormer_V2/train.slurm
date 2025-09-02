#!/bin/bash

#SBATCH --job-name=policy_train         # 잡 이름 (Job name)
#SBATCH --output=logs/%x_%j.out     # 표준 출력 로그 파일 (%x: job-name, %j: jobid)
#SBATCH --error=logs/%x_%j.err      # 에러 로그 파일
#SBATCH --partition=suma_rtx4090        # 사용할 파티션 이름
#SBATCH --gres=gpu:1                # GPU 1개 요청 (클러스터 설정에 맞게)
#SBATCH --time=24:00:00             # 실행 시간 제한 (HH:MM:SS)
# 그 외

module swap cuda/12.5 cuda/12.1
source /home/meat124/anaconda3/bin/activate RoboTwin_DiT_test
conda activate RoboTwin_DiT_test              # Conda 환경 활성화
echo "Conda environment activated."

# 실제 실행 명령
echo "Starting training script in DiT_Graphormer_V2 directory."
# bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id}
srun bash train.sh stack_blocks_three demo_clean 100 0 0
echo "Training script completed."