
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
from datetime import datetime

epochs = 20
        
def main():
    config = config_LSS()
    config.initialize(root_dir=['/home/workspace/logs/dataset'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained_model_dir = None
    state_dict_file = 'model_0020.pth'
    model_save_dir = '/home/workspace/pretrained_models/LSS_tmp' + '/' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model = LiftSplatShoot(config)
    if pretrained_model_dir is not None:
        state_dict = torch.load(os.path.join(pretrained_model_dir, state_dict_file), map_location=device)
        model.load_state_dict(state_dict, strict=True)

    
    train_set = CARLA_Data(root=config.data_roots,
                        config=config,
                        estimate_class_distributions=config.estimate_class_distributions,
                        estimate_sem_distribution=config.estimate_semantic_distribution,
                        validation=False)

    model_wrapper = LSS_wrapper(model, config, device)

    start_epoch = 0  # Epoch to continue training from
    optimizer = optim.Adam(model_wrapper.parameters(), lr=1e-4, weight_decay=1e-7)

    val_set = CARLA_Data(root=config.data_roots, config=config, validation=True)

    dataloader_train = DataLoader(train_set,
                                batch_size=6,
                                shuffle = True,
                                num_workers = 4,
                                pin_memory=True)
    
    dataloader_val = DataLoader(val_set,
                                 batch_size=6,
                                 shuffle = False,
                                num_workers = 4,
                                pin_memory=True)
    
    trainer = Engine(model_wrapper=model_wrapper,
                    optimizer=optimizer,
                    dataloader_train=dataloader_train,
                    dataloader_val=dataloader_val,
                    config=config,
                    device=device,
                    cur_epoch=start_epoch)
    
    torch.backends.cudnn.benchmark = True
    
    for epoch in range(trainer.cur_epoch, epochs):
        print(f'-----------Epoch {epoch}----------')
        trainer.train()
        trainer.validate()
        trainer.plot_loss()
        trainer.cur_epoch += 1

    os.makedirs(model_save_dir, exist_ok=True)
    trainer.save(model_save_dir)
    plt.pause()

if __name__ == '__main__':
    main()