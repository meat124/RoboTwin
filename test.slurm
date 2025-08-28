#!/bin/bash

#SBATCH --job-name=convert_data         # 잡 이름 (Job name)
#SBATCH --output=logs/%x_%j.out     # 표준 출력 로그 파일 (%x: job-name, %j: jobid)
#SBATCH --error=logs/%x_%j.err      # 에러 로그 파일
#SBATCH --partition=base_suma_rtx3090        # 사용할 파티션 이름
#SBATCH --gres=gpu:1                # GPU 1개 요청 (클러스터 설정에 맞게)
#SBATCH --time=3:00:00             # 실행 시간 제한 (HH:MM:SS)

module swap cuda/12.5 cuda/12.1
source /home/meat124/anaconda3/bin/activate RoboTwin_DiT_MoGe
conda activate RoboTwin_DiT_MoGe
echo "Conda environment activated."

echo "Data conversion started"
cd policy/DiT
srun python convert_robotwin_to_dit.py \
    --data_dir /scratch2/meat124/dit_ws/src/RoboTwin/data/place_dual_shoes/demo_clean/data \
    --out_dir /scratch2/meat124/dit_ws/src/RoboTwin/data_dit/place_dual_shoes/demo_clean