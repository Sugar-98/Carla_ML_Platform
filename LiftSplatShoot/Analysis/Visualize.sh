#!/bin/bash

#====path settings====
source ../../env.sh #Define global path settings

data_path="$PROJECT_ROOT/logs/dataset_plot"
pretrained_model_dir="$PROJECT_ROOT/pretrained_models/LSS/2026-01-31_02-22-05" #Path for pretrained model. 
state_dict_file='model_latest.pth'  #Pretrained model file
save_path="$PROJECT_ROOT/pretrained_models/LSS/2026-01-31_02-22-05/visualize.mp4"

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/analyze_model.py" #Define train script

(
  cd "$WORK_DIR" || exit

  python3 "$SCRIPT_PATH" \
    --data "$data_path" \
    --save_path "$save_path" \
    --state_dict_file "$state_dict_file" \
    --pretrained_model_path "$pretrained_model_dir"

)