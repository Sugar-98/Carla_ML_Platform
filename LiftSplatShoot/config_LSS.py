from common.config.Train_conf import Train_conf
from common.utils import cal_mfb_weights
import numpy as np
from pathlib import Path
from datetime import datetime

class config_LSS(Train_conf):
  def __init__(self):
    super().__init__()

    #model parameters------------------------------------------
    xbound=[-32.0, 32.0, 0.25]
    ybound=[-32.0, 32.0, 0.25]
    zbound=[-10.0, 10.0, 20.0]
    dbound=[4.0, 32.0, 0.5]
    
    self.grid_conf = {
      'xbound': xbound,
      'ybound': ybound,
      'zbound': zbound,
      'dbound': dbound,
    }

    self.outC = 11
    
    #Training parameters------------------------------------------
    self.epochs = 20  # Number of epochs to train
    self.batch_size = 6  # Batch size used during training
    self.lr = 1e-4  # Learning rate used for training
    self.weight_decay = 1e-7  # Weight decay coefficient used during training
    
    self.ignore_class = []
    hist = np.array([1.0510e+08, 1.5803e+07, 9.9302e+06, 8.3205e+06, 2.4915e+06,
                    6.2880e+03, 2.1756e+04, 6.2400e+02, 3.5760e+04, 2.8545e+06, 5.4320e+03])

    self.bev_semantic_weights = cal_mfb_weights(hist, ignore_class=self.ignore_class)

  def initialize(self, root_dir='', setting='all', **kwargs):
    super().initialize(root_dir=root_dir, setting=setting, **kwargs)
    self.DataLoader_config.ignore_class = self.ignore_class
    self.DataLoader_config.use_post_augment = True #Post augmentation for RGB
    self.DataLoader_config.use_bev_post_augment = True #Post augmentation for BEV
    self.DataLoader_config.num_max_data_train = 80000
    self.DataLoader_config.num_max_data_val = 5000
    self.val_towns = [13]