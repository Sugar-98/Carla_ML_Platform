import torch
from torch.utils.data import DataLoader

from config_LSS import config_LSS
from train_lib.data import CARLA_Data
from tqdm import tqdm

epochs = 1
  
def main():
  config = config_LSS()
  config.initialize(root_dir=['/home/workspace/logs/dataset_tmp/scenario'])

  val_set = CARLA_Data(root=config.data_roots, config=config, validation=False)
  
  dataloader_val = DataLoader(val_set,
        batch_size=1,
        shuffle = False,
        num_workers = 0)

  hist_bev = torch.zeros(1,11)
  with torch.no_grad():
    for data in tqdm(dataloader_val):
      label_bev_semantics = data["bev_semantic"]
      hist_bev_tmp = torch.bincount(label_bev_semantics[0,:,:].flatten(), minlength=11)
      hist_bev += hist_bev_tmp

  eps = 1e-6
  p = hist_bev.float() / (hist_bev.sum().float() + eps)  # クラス出現確率
  print(hist_bev)

if __name__ == '__main__':
  main()