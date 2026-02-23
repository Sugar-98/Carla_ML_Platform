
import torch
from torch import optim
import os
from torch.utils.data import DataLoader
from config_LSS import config_LSS
from train_lib.data import CARLA_Data
from LiftSplatShoot import LiftSplatShoot
from train_lib.train_engine import Engine
from LSS_wrapper import LSS_wrapper
from datetime import datetime
import argparse
import sys
from config import GlobalConfig
from common.config.DataLoader_conf import DataLoader_conf
from common.config.DataAgent_conf import DataAgent_conf
from common.utils import MP4Writer
import cv2
import time
import json


class SaveMode:
  VIDEO_ONLY = 0
  IMAGES_ONLY = 1
  BOTH = 2

def parse_args():
  parser = argparse.ArgumentParser(
    description="Plot LSS"
  )

  parser.add_argument("--data", type=str, required=True,
            help="Path to dataset")
  
  parser.add_argument("--pretrained_model_path", type=str,
            help="Path for pretrained model. Model is traind from initial state when None")
  
  parser.add_argument("--state_dict_file", type=str,
            help="File name of pretrained model")

  parser.add_argument("--save_path", type=str, default=None,
            help="Base directory for saving outputs")

  parser.add_argument("--save_mode", type=int, default=SaveMode.VIDEO_ONLY,
            choices=[SaveMode.VIDEO_ONLY, SaveMode.IMAGES_ONLY, SaveMode.BOTH],
            help="0: video only, 1: images only, 2: both")
  
  return parser.parse_args()


    
def main():
  args = parse_args()
  
  data_path = args.data
  pretrained_model_dir = args.pretrained_model_path
  state_dict_file = args.state_dict_file
  save_path = args.save_path
  save_mode = args.save_mode

  visualize_dir = None
  if save_path is not None:
    visualize_dir = os.path.join(save_path, "visualize")
    os.makedirs(visualize_dir, exist_ok=True)

  carla_garage_config = GlobalConfig()
  DataAgent_config = DataAgent_conf()
  DataLoader_config = DataLoader_conf(DataAgent_config)

  config = config_LSS()
  config.initialize(root_dir=[data_path],
            carla_garage_config = carla_garage_config,
            DataAgent_config = DataAgent_config, 
            DataLoader_config = DataLoader_config
            )
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = LiftSplatShoot(config)
  state_dict = torch.load(os.path.join(pretrained_model_dir, state_dict_file), map_location=device)
  model.load_state_dict(state_dict, strict=True)

  model_wrapper = LSS_wrapper(model, config, device)

  val_set = CARLA_Data(root=config.data_roots,
            DataLoader_config=config.DataLoader_config,
            carla_garage_config = carla_garage_config,
            DataAgent_config = config.DataAgent_config, 
            validation=True,
            val_towns = config.val_towns)
  
  dataloader_val = DataLoader(val_set,
                 batch_size=1,
                 shuffle = False,
                num_workers = 4,
                pin_memory=True)
  
  torch.backends.cudnn.benchmark = True

  save_video = visualize_dir is not None and save_mode in (SaveMode.VIDEO_ONLY, SaveMode.BOTH)
  save_images = visualize_dir is not None and save_mode in (SaveMode.IMAGES_ONLY, SaveMode.BOTH)

  if save_video:
    video = MP4Writer(
        save_path=os.path.join(visualize_dir, "output.mp4"),
        fps=10
    )
  
  for frame_idx, data in enumerate(dataloader_val):
    model_wrapper.load_data_compute_loss(data)
    img = model_wrapper.plot_model_out()
    if save_video:
      video.write(img)
    if save_images:
      img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      cv2.imwrite(os.path.join(visualize_dir, f"frame_{frame_idx:05d}.png"), img_bgr)

  if save_video:
    video.close()

  print("Press any key to continue...")
  if sys.platform.startswith("win"):
    import msvcrt
    msvcrt.getch()
  else:
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
      tty.setraw(fd)
      sys.stdin.read(1)
    finally:
      termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == '__main__':
  main()