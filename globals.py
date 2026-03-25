
import numpy as np
import re
from pathlib import Path


# ============================================================
# Global filename pattern and plotting defaults
# ============================================================
_FILENAME_PATTERN = re.compile(
    r"(?P<fluence>[-+]?\d*.?\d+)fs"
    r"hw(?P<delay>[-+]?\d*.?\d+(?:e[-+]?\d+)?)"
    r"delay(?P<image_number>\d+).tif$",
    re.IGNORECASE,
)

FIGSIZE = (10, 4)
STD_FACTOR = 3
MAX_PROCESSORS = 4
# ============================================================
# Global detector / beam configuration
# Edit these defaults for your instrument
# ============================================================
CENTER_X = 746
CENTER_Y = 1621
PIXEL1 = 1.72e-4                 # m, detector pixel size along rows (y)
PIXEL2 = 1.72e-4                 # m, detector pixel size along cols (x)
DISTANCE = 0.1885              # m, sample-to-detector distance
WAVELENGTH = 1.0596937286439283e-10          # m

# Detector orientation
TILT_ANGLE = np.deg2rad(-0.16433)               # rad
TILT_PLANE_ROTATION = np.deg2rad(72.12779)      # rad
ROT3 = 0.0                      # rad, in-plane detector rotation

# Optional corrections
POLARIZATION_FACTOR = None      # e.g. 0.99 or None
DARK = None                     # 2D dark image or None
FLAT = None                     # 2D flat-field image or None

# ============================================================
# Form Factor CSV File
# ============================================================
FORM_FACTOR_FILE = Path(
    r"C:\Users\lheald\Documents\Guzelturk_Lab\gued\packages\x_ray_ff\atomic_FF_coeffs_clean.csv"
)