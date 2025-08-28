#!/bin/bash

module swap cuda/12.5 cuda/12.1
source /home/meat124/anaconda3/bin/activate RoboTwin_DiT_MoGe
conda activate RoboTwin_DiT_MoGe
echo "Conda environment activated."

echo "Data conversion started"
cd policy/DiT
srun python convert_robotwin_to_dit.py \
    --data_dir /scratch2/meat124/dit_ws/src/RoboTwin/data/place_dual_shoes/demo_clean/data \
    --out_dir /scratch2/meat124/dit_ws/src/RoboTwin/data_dit/place_dual_shoes/demo_clean