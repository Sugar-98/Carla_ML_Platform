
import torch
from torch.utils.data import Dataset
from torch import optim
from imgaug import augmenters as ia
import re
import os
import gzip
import jsonpickle
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config_LSS import config_LSS
from train_lib.data import CARLA_Data
from tqdm import tqdm

from LiftSplatShoot import LiftSplatShoot
from train_lib.train_engine import Engine
from LSS_wrapper import LSS_wrapper

epochs = 1
    
def main():
  config = config_LSS()
  config.initialize(root_dir=['/home/workspace/logs/dataset'])
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


  model_file_dir = '/home/workspace/pretrained_models/LSS_tmp/2025-11-04_13-19-37'
  model = LiftSplatShoot(config)

  files = [file for file in os.listdir(model_file_dir) if file.endswith('.pth') and file.startswith('model')]
  assert files, f"No .pth files found in {model_file_dir}"
  file = files[0]

  state_dict = torch.load(os.path.join(model_file_dir, file), map_location=device)
  model.load_state_dict(state_dict, strict=True)

  model_wrapper = LSS_wrapper(model, config, device)

  val_set = CARLA_Data(root=config.data_roots, config=config, validation=True)
  
  dataloader_val = DataLoader(val_set,
                 batch_size=1,
                 shuffle = False,
                num_workers = 4)
  
  model_wrapper.eval()

  for epoch in range(epochs):
    print(f'-----------Epoch {epoch}----------')
    if len(dataloader_val) != 0:
      with torch.no_grad():
        for data in tqdm(dataloader_val):
          pred, label, loss, tmp_loss_individual = model_wrapper.load_data_compute_loss(data)
          model_wrapper.plot_model_out()


  plt.pause()

if __name__ == '__main__':
  main()