import logging
import os
import time

import numpy as np
import torch
import torch as tc

from common.pose_nets.config import cfg


def initLogging(logFilename):
    """Init for logging"""
    logger = logging.getLogger("")

    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s-%(levelname)s] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
            filename=logFilename,
            filemode="w",
        )
        console = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s-%(levelname)s] %(message)s")
        console.setFormatter(formatter)
        logger.addHandler(console)

def init(cfg):
    taskfolder = f"./results/"
    if not os.path.exists(taskfolder):
        os.makedirs(taskfolder)
    datafmt = time.strftime("%Y%m%d_%H%M%S")

    log_dir = f"{taskfolder}/{cfg.dataset}_{datafmt}.log"
    initLogging(log_dir)
    ckpt_path = f"{taskfolder}/{cfg.dataset}_{datafmt}.pt"
    return ckpt_path
