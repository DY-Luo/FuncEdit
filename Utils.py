# Utils

import os
import scipy
import numpy as np
import pandas as pd

def tp10k_transform(DGE,norm_factor=1.0e4):
    """normalize columns of pandas dataframe to sum to a constant, by default 10,000"""
    return(norm_factor*(DGE / DGE.sum()))



