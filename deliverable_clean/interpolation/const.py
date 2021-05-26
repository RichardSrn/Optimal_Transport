"""
STORING CONSTANTS HERE.
"""

import os

from ot_emd import ot_emd
from ROT import ROT


CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = CURR_DIR + "/artificial_data.npy"

TEST_ALG = ot_emd
# TEST_ALG = ROT

PLOT_TITLE = f"Testing {TEST_ALG}"
