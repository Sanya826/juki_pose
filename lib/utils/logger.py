import os
import sys

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self,log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_log(self, log_dict):
        for k,v in log_dict.items():
            self.writer.add_scalar(k, v[0], v[1])
    
    def close(self):
        self.writer.close()


