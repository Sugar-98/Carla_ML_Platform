import argparse
import torch
from torch import optim
import os
from torch.utils.data import DataLoader
from LiftSplatShoot.config_LSS import config_LSS
from train_lib.data import CARLA_Data
import sys
from config import GlobalConfig
from common.config.DataLoader_conf import DataLoader_conf
from common.config.DataAgent_conf import DataAgent_conf
from common.config.DataAnalysis_conf import DataAnalysis_conf
from pathlib import Path
from train_lib.dataset_analyzer import Analyzer


def parse_args():
  parser = argparse.ArgumentParser(
    description="Analyze dataset distribution and statistics."
  )
  parser.add_argument("--data", type=str, required=True,
            help="Path to dataset directory")
  parser.add_argument("--save_path", type=str, required=True,
            help="Path to save analysis results")
  return parser.parse_args()


def main():
  args = parse_args()
  data_path = args.data
  save_pth = args.save_path

  carla_garage_config = GlobalConfig()
  DataAgent_config = DataAgent_conf()
  DataLoader_config = DataLoader_conf(DataAgent_config)

  config = config_LSS()
  config.initialize(root_dir=[data_path],
            carla_garage_config = carla_garage_config,
            DataAgent_config = DataAgent_config, 
            DataLoader_config = DataLoader_config
            )
  
  train_set = CARLA_Data(root=config.data_roots,
            DataLoader_config=config.DataLoader_config,
            carla_garage_config = carla_garage_config,
            DataAgent_config = config.DataAgent_config, 
            validation=False,
            val_towns = config.val_towns)

  val_set = CARLA_Data(root=config.data_roots,
            DataLoader_config=config.DataLoader_config,
            carla_garage_config = carla_garage_config,
            DataAgent_config = config.DataAgent_config, 
            validation=True,
            val_towns = config.val_towns)

  dataloader_train = DataLoader(train_set,
                batch_size=6,
                shuffle = False,
                num_workers = 6,
                pin_memory=True)
  
  dataloader_val = DataLoader(val_set,
                 batch_size=6,
                 shuffle = False,
                num_workers = 6,
                pin_memory=True)
  
  analyzer = Analyzer(
    dataloader_train,
    dataloader_val,
    len(train_set),
    len(val_set),
    carla_garage_config,
    analysis_config=DataAnalysis_conf()
  )

  analyzer(save_pth)

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