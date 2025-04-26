import sys
import torch


DEFAULT_SEED = 666
C_SEARCH_GRID = 1 + 10.**torch.arange(-4, 9, 2)
