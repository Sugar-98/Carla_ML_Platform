#!/bin/bash

#====path settings====
source ../../env.sh #Define global path settings

data_path="$PROJECT_ROOT/logs/dataset_plot"
pretrained_model_dir="$PROJECT_ROOT/pretrained_models/LSS" #Path for pretrained model. 
state_dict_file='model_best.pth'  #Pretrained model file
save_path="$PROJECT_ROOT/pretrained_models/LSS"

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_LSS.py" #Define train script

(
  cd "$WORK_DIR" || exit

  python3 "$SCRIPT_PATH" \
    --data "$data_path" \
    --save_path "$save_path" \
    --state_dict_file "$state_dict_file" \
    --pretrained_model_path "$pretrained_model_dir"

)