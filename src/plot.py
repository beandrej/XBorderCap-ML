import os
import sys; sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.plot_utils import *

for border in config.FBMC_BORDERS:
    plotActualVsPredicted('FX_FBMC_NORM', 'Hybrid', border)