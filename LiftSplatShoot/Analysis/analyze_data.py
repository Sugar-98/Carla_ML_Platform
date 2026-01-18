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
from pathlib import Path
from train_lib.dataset_analyzer import Analyzer

def main():
  root_dir = Path(__file__).resolve().parent.parent.parent
  data_path = f"{root_dir}/logs/dataset"
  save_pth = f"{root_dir}/analysis"

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
    carla_garage_config
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