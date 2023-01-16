import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import random
import numpy as np
import scipy
import pandas as pd
from tqdm import tqdm
from engine import train_test
from utils import  read_preprocess_data

logger = logging.getLogger(__name__)

@hydra.main(version_base=None , config_path = "conf",config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    logger.info("Starting ....")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    train_ftrs, test_ftrs = read_preprocess_data(cfg)
    test_df = train_test(train_ftrs, test_ftrs, cfg)
    test_df.to_csv(f'{cfg.input_dir}/submission.csv', index = False)
    

if __name__ == "__main__":
    main()