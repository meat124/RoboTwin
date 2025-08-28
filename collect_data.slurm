#!/bin/bash

module swap cuda/12.5 cuda/12.1
source /home/meat124/anaconda3/bin/activate RoboTwin_DiT_test
conda activate RoboTwin_DiT_test
echo "Conda environment activated."

echo "Data collection started"
srun bash collect_data.sh handover_block demo_clean 0