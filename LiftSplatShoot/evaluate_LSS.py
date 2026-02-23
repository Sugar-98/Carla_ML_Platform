import torch
from torch.utils.data import DataLoader
from config_LSS import config_LSS
from train_lib.data import CARLA_Data
from LiftSplatShoot import LiftSplatShoot
from LSS_wrapper import LSS_wrapper
from datetime import datetime
import argparse
import os
import pandas as pd
from config import GlobalConfig
from common.config.DataLoader_conf import DataLoader_conf
from common.config.DataAgent_conf import DataAgent_conf
import time
from train_lib.train_engine import Engine  # Import the Engine class

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LSS"
    )

    # Path to the dataset
    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset")

    # Path to save evaluation logs
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save evaluation logs")

    # Path to the pretrained model directory
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="Path for pretrained model")

    # File name of the pretrained model
    parser.add_argument("--state_dict_file", type=str, required=True,
                        help="File name of pretrained model")

    return parser.parse_args()

# Main function to perform evaluation
def main():
    tic = time.perf_counter()
    args = parse_args()

    # Parse arguments
    data_path = args.data
    pretrained_model_dir = args.pretrained_model_path
    state_dict_file = args.state_dict_file
    save_path = args.save_path

    # Create directory to save evaluation logs
    os.makedirs(save_path, exist_ok=True)

    # Initialize configurations
    carla_garage_config = GlobalConfig()
    DataAgent_config = DataAgent_conf()
    DataLoader_config = DataLoader_conf(DataAgent_config)

    config = config_LSS()
    config.initialize(root_dir=[data_path],
                      carla_garage_config=carla_garage_config,
                      DataAgent_config=DataAgent_config,
                      DataLoader_config=DataLoader_config
                      )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pretrained model
    model = LiftSplatShoot(config)
    state_dict = torch.load(os.path.join(pretrained_model_dir, state_dict_file), map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # Prepare validation dataset
    val_set = CARLA_Data(root=config.data_roots,
                         DataLoader_config=config.DataLoader_config,
                         carla_garage_config=carla_garage_config,
                         DataAgent_config=config.DataAgent_config,
                         validation=True,
                         val_towns=config.val_towns)

    # Create DataLoader for validation dataset
    dataloader_val = DataLoader(val_set,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 pin_memory=True)

    # Wrap the model for evaluation
    model_wrapper = LSS_wrapper(model, config, device)

    # Initialize Engine with only validation DataLoader
    engine = Engine(model_wrapper=model_wrapper,
                    optimizer=None,  # No optimizer needed for evaluation
                    dataloader_train=None,  # No training DataLoader
                    dataloader_val=dataloader_val,
                    Train_config=config,
                    device=device)

    # Perform validation
    engine.validate()

    # Save evaluation results to log.csv
    engine.save_log(save_path)

    # Print evaluation completion message
    toc = time.perf_counter() - tic
    print(f"Evaluation completed in {toc / 60:.2f} minutes. Logs saved to {os.path.join(save_path, 'log.csv')}.")


if __name__ == '__main__':
    main()