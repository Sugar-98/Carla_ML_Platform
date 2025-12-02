
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LSS"
    )

    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset")
    
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to model save dir")
    
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path for pretrained model. Model is traind from initial state when None")
    
    parser.add_argument("--state_dict_file", type=str, default=None,
                        help="File name of pretrained model")
    
    return parser.parse_args()
        
def main():
    args = parse_args()

    data_path = args.data
    pretrained_model_dir = args.pretrained_model_path
    state_dict_file = args.state_dict_file
    model_save_dir = args.save_path + '/' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    config = config_LSS()
    config.initialize(root_dir=[data_path])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    for epoch in range(trainer.cur_epoch, config.epochs):
        print(f'-----------Epoch {epoch}----------')
        trainer.train()
        trainer.validate()
        trainer.plot_loss()
        trainer.cur_epoch += 1

    os.makedirs(model_save_dir, exist_ok=True)
    trainer.save(model_save_dir)
    
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