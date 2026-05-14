"""
This file loads in diffraction images which are saved as raw .tif files, applies a beam stop mask, 
applies a preset detector mask, performs azimuthal averaging, normalizes the azimuthal data, subtracts
a background profile based on a specified background image, and saves each azimuthally averaged, 
background subtracted profile as a .dat file in the specified output directory. The .dat files 
are saved with a filename that includes the fluence and delay extracted from the original .tif filename, 
e.g. "BTO400nmS3_240Ksurv3_550fs_hw0.6ns_delay00001.dat". The .dat files contain two columns: the first 
column is the q values (in inverse Angstroms) and the second column is the normalized, 
background-subtracted intensity values. The code is designed to be run in batch mode, processing all 
relevant .tif files in the specified data directory and saving the processed .dat files to the 
specified output directory. 
"""

from pathlib import Path
import numpy as np

import trxrd

# Experimental Parameters and Defaults
# ============================================================
# Data and Mask Paths, Scan Name, and Filename Pattern
# ============================================================
DATA_PATH = Path(r"\\s7data\beams46\7IDC\Cotts\2025_11Exp\BTO400_S3") # Path to directory containing TIFF files
MASK_FILE = Path(r"C:\Users\lheald\Documents\Guzelturk_Lab\TRXRDPython\testdata\mask_2021_dec.tif") # Path to mask file
SCAN_NAME = "BTO400nmS3_360Kre4" # Prefix in file name to identify relevant files, e.g. "550nm_re" etc.
SCAN_TYPE = "delay_scan" # Type of scan based on filename pattern, e.g. "delay_scan", "theta_samz", etc. Must correspond to a key in the "filename_patterns" dictionary in trxrd.py
BACKGROUND_PATH = Path(r"\\s7data\beams46\7IDC\Cotts\2025_11Exp\BlankSubstratePinkBeam\blanksubstratePinkBeam285K-1.0fshw-4e-09delay00004_045.tif")
SAVE_PATH = Path(r"C:\Users\lheald\Documents\Guzelturk_Lab\Cotts_Processed_Data\BTO400nmS3_360Kre4") # Path to directory where processed data will be saved, e.g. as .h5 file

# ============================================================
# General Defaults
# ============================================================
FIGSIZE = (10, 4)
STD_FACTOR = 3
MAX_PROCESSORS = 4
DELAY_SIGN = -1 # Check file naming scheme, sometimes positives delays have "-" in front and negative have no sign so need to invert sign

# ============================================================
# Beam Stop Mask Defaults
# ============================================================
MASK_CENTER_X = 52
MASK_CENTER_Y = 1667
MASK_RADIUS = 30

# ============================================================
# Center Guess and Sampling Defaults
# ============================================================
CENTER_X = 44
CENTER_Y = 1666
DOWNSAMPLE = 2 # Downsample factor for center finding, e.g. 2 means use every other pixel, 4 means use every 4th pixel, etc.

# ============================================================
# Detector Parameters and Defaults
# ============================================================
# PONI file from pyFAI-calib2. Set to a Path to use PONI geometry;
# set to None to use the manual parameters below.
PONI_FILE = Path(r"C:\Users\lheald\Documents\Guzelturk_Lab\Cotts_Processed_Data\CeO2\low_keV\CeO2_poni.poni")                # Path to .poni file, or None
# Detector and beam parameters
PIXEL1 = 1.72e-4                 # m, detector pixel size along rows (y)
PIXEL2 = 1.72e-4                 # m, detector pixel size along cols (x)
DISTANCE = 0.1723                 # m, sample-to-detector distance
WAVELENGTH = 0.39738514824147314e-10          # m

# Detector orientation
TILT_ANGLE = np.deg2rad(0)               # rad
TILT_PLANE_ROTATION = np.deg2rad(90)      # rad
ROT3 = 0.0                      # rad, in-plane detector rotation

# Optional corrections
POLARIZATION_FACTOR = 0.999      # e.g. 0.99 or None
DARK = None                     # 2D dark image or None
FLAT = None                     # 2D flat-field image or None

# ============================================================
# Azimuthal Averaging and Normalization Defaults
# ============================================================
UNIT = "q_A^-1" # Unit for x-axis of azimuthally averaged data, e.g. "q_A^-1" for inverse Angstroms, "2theta_deg" for degrees, etc. 
NAN_MIN = 0.35 # Minimum value for valid data, values below this will be set to NaN, e.g. 0.35 or None for no minimum threshold
NAN_MAX = None # Maximum value for valid data, values above this will be set to NaN, e.g. 1.0 or None for no maximum threshold
NORM_MIN = 1.25 # Minimum value for normalization, values below this will be set to this value before normalization, e.g. 0.5 or None for no minimum threshold
NORM_MAX = 1.50 # Maximum value for normalization, values above this will be set to this value before normalization, e.g. 1.0 or None for no maximum threshold
N_POINTS = 3000 # Number of points for azimuthal averaging, e.g. 3000 or None to use all pixels



# Check number of files in folder 
file_names = sorted(DATA_PATH.glob(f"{SCAN_NAME}*.tif"))
print(f"{len(file_names)} TIFF files found in {DATA_PATH}.")


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
data_dict = trxrd.get_images_by_scan_name(
    folder_path=DATA_PATH,
    scan_name=SCAN_NAME,
    sort=True,
    filter_data=False,
    plot=False,
)

print(data_dict.keys())
print("Images shape:", data_dict["images"].shape)
print("Counts shape:", data_dict["counts"].shape)

# ------------------------------------------------------------
# Build Masks
# ------------------------------------------------------------
image_shape = data_dict["images"].shape[1:]   # (rows, cols)

combined_mask = trxrd.build_combined_mask(
    image_shape=image_shape,
    center_xy=(MASK_CENTER_X, MASK_CENTER_Y),
    radius=MASK_RADIUS,
    mask_path=MASK_FILE,
)

# ------------------------------------------------------------
# Compute azimuthal average
# ------------------------------------------------------------
if PONI_FILE is not None:
    az_result = trxrd.azimuthal_average_pyfai(
        images=data_dict["images"],
        poni_path=PONI_FILE,
        npt=N_POINTS,
        unit=UNIT,
        nan_radial_range=(NAN_MIN, NAN_MAX),   # set Q < 0.3 to NaN
        azimuth_range=None,
        integration_mask=combined_mask,
        return_dict=True,
        progress_interval=100,
        use_custom_polarization=False,
        integration_function="integrate1d",
        correct_solid_angle=False,
        method=("bbox", "csr", "cython")
    )
else:
    az_result = trxrd.azimuthal_average_manual(
        images=data_dict["images"],
        center_xy=(CENTER_X, CENTER_Y),
        pixel_size=(PIXEL1, PIXEL2),
        distance=DISTANCE,
        wavelength=WAVELENGTH,
        tilt_angles=(TILT_ANGLE, TILT_PLANE_ROTATION, ROT3),
        npt=N_POINTS,
        unit=UNIT,
        nan_radial_range=(NAN_MIN, NAN_MAX),   # set Q < 0.3 to NaN
        azimuth_range=None,
        integration_mask=combined_mask,
        return_dict=True,
        progress_interval=100,
        polarization_factor=POLARIZATION_FACTOR,
        dark=DARK,
        flat=FLAT,
    )

q = az_result["radial"]
profiles = az_result["profiles"]


# ------------------------------------------------------------
# Normalize profiles
# ------------------------------------------------------------
norm_result = trxrd.normalize_profiles_to_range(
    radial=q,
    profiles=profiles,
    norm_range=(NORM_MIN, NORM_MAX),   # example
    mode="mean",
    plot=False,
    show_normalized_plot=False,
    return_dict=True,
    plot_factors=False,
)

profiles_norm = norm_result["normalized_profiles"]


# ------------------------------------------------------------
# Save profiles as .dat files
# ------------------------------------------------------------

SAVE_PATH.mkdir(parents=True, exist_ok=True)

# delays = data_dict["delay"]
# # If you also want fluence, include it if available:
# fluence = data_dict.get("fluence", None)

for i, profile in enumerate(profiles_norm):

    if not np.any(np.isfinite(profile)):
        print(f"Skipping index {i} (all NaN)")
        continue

    input_file = Path(data_dict["file_names"][i])
    filename = input_file.with_suffix(".dat").name
    output_file = SAVE_PATH / filename

    valid_mask = np.isfinite(q) & np.isfinite(profile)
    data_to_save = np.column_stack((q[valid_mask], profile[valid_mask]))

    header = "q (A^-1)\tI_normalized (a.u.)"

    np.savetxt(
        output_file,
        data_to_save,
        header=header,
        comments="",
    )

    print(f"Saved: {output_file.name}")