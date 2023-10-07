import math
from typing import Optional

import torch
from torch import Tensor
from torch import nn
import numpy as np
import torch.nn.functional as F



class RegressionHead(torch.nn.Module):
    def __init__(self, input_dim):
        super(RegressionHead, self).__init__()
        self.fc = torch.nn.Linear(input_dim, 1)  # Map input_dim to a single scalar
        
    def forward(self, x):
        return self.fc(x)