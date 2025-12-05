#!/bin/bash

#====path settings====
source ../env.sh #Define global path settings

data_path="$PROJECT_ROOT/logs/dataset"
pretrained_model_dir="" #Path for pretrained model. Model is traind from initial state when "None"
state_dict_file='model_0020.pth'  #Pretrained model file
model_save_dir="$$PROJECT_ROOT/pretrained_models/LSS"

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_LSS.py" #Define train script

(
  cd "$WORK_DIR" || exit

  if [ -n "$pretrained_model_dir" ]; then
    python3 "$SCRIPT_PATH" \
      --data "$data_path" \
      --save_path "$model_save_dir" \
      --state_dict_file "$state_dict_file" \
      --pretrained_model_path "$pretrained_model_dir"
  else
    python3 "$SCRIPT_PATH" \
      --data "$data_path" \
      --save_path "$model_save_dir"
  fi
)