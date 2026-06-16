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
import tifffile as tf

import trxrd

# Experimental Parameters and Defaults

# ============================================================
# Data and Mask Paths, Scan Name, and Filename Pattern
# ============================================================
DATA_PATH = Path(r"\\s7data\beams46\7IDC\Cotts\2025_11Exp\18keV\BTO400S3_18keV") # Path to directory containing TIFF files
MASK_FILE = Path(r"C:\Users\lheald\Documents\Guzelturk_Lab\TRXRDPython\testdata\mask_2021_dec.tif") # Path to mask file
SCAN_NAME = "BTO400nmS3_360Klong" # Prefix in file name to identify relevant files, e.g. "550nm_re" etc.
SCAN_TYPE = "delay_scan" # Type of scan based on filename pattern, e.g. "delay_scan", "theta_samz", etc. Must correspond to a key in the "filename_patterns" dictionary in trxrd.py
BACKGROUND_PATH = Path(r"\\s7data\beams46\7IDC\Cotts\2025_11Exp\18keV\BlankSubstrate\BlankSubstrate_1p0theta-1.0fshw2e-08delay00007_9578.tif")
SAVE_PATH = Path(r"C:\Users\lheald\Documents\Guzelturk_Lab\Cotts_Processed_Data\BTO_18keV\BTO400nmS3_360Klong") # Path to directory where processed data will be saved, e.g. as .h5 file
# PONI file from pyFAI-calib2. Set to a Path to use PONI geometry;
# set to None to use the manual parameters below.
PONI_FILE = Path(r"C:\Users\lheald\Documents\Guzelturk_Lab\Cotts_Processed_Data\CeO2_18keV\CeO2_18keV.poni") # Path to .poni file, or None

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
MASK_RADIUS = 20

# ============================================================
# Center Guess and Sampling Defaults
# ============================================================
CENTER_X = 44
CENTER_Y = 1666
DOWNSAMPLE = 2 # Downsample factor for center finding, e.g. 2 means use every other pixel, 4 means use every 4th pixel, etc.

# ============================================================
# Detector Parameters and Defaults
# ============================================================

#PONI_FILE = None
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
NORM_MIN = 4.05 # Minimum value for normalization, values below this will be set to this value before normalization, e.g. 0.5 or None for no minimum threshold
NORM_MAX = 4.25 # Maximum value for normalization, values above this will be set to this value before normalization, e.g. 1.0 or None for no maximum threshold
N_POINTS = 3000 # Number of points for azimuthal averaging, e.g. 3000 or None to use all pixels

# if META_PATH is not None:
#     metadata = trxrd.read_tif_metadata(META_PATH)
#     #print(metadata.keys())
#     CENTER_X = metadata["center_x"]
#     MASK_CENTER_X = metadata["center_x"]
#     CENTER_Y = metadata["center_y"]
#     MASK_CENTER_Y = metadata["center_y"]
#     DISTANCE = metadata["distance"]
#     WAVELENGTH = metadata["wavelength"]
#     POLARIZATION_FACTOR = metadata["polarization"]
#     PIXEL1 = 1.50e-4
#     PIXEL2 = 1.50e-4

# # Check number of files in folder 
# file_names = sorted(DATA_PATH.glob(f"{SCAN_NAME}*.tif"))
# print(f"{len(file_names)} TIFF files found in {DATA_PATH}.")


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
# data_dict = trxrd.get_images_by_scan_name(
#     folder_path=DATA_PATH,
#     scan_name=SCAN_NAME,
#     sort=True,
#     filter_data=False,
#     plot=False,
# )

# data_dict = trxrd.get_image_details(
#     folder_path=DATA_PATH,
#     sample_name=SCAN_NAME,
#     filename_scheme="delay_scan",
#     sort_key="image_number",
#     sort=True,
#     filter_data=False,
#     plot=False,
# )

# print(data_dict.keys())
# print("Images shape:", data_dict["images"].shape)
# print("Counts shape:", data_dict["counts"].shape)

# ------------------------------------------------------------
# Define File Paths
# ------------------------------------------------------------
# Main diffraction data folder
data_path = DATA_PATH

# Check number of files in folder 
file_names = sorted(data_path.glob(f"{SCAN_NAME}-*.tif"))
print(f"{len(file_names)} TIFF files found in {data_path}.")

# Detector mask file
mask_file = MASK_FILE
print(f"Using mask file: {mask_file}")

sample_files = sorted(DATA_PATH.glob(f"{SCAN_NAME}-*.tif"))
sample_image = tf.imread(str(sample_files[0]))


# ------------------------------------------------------------
# Build Masks
# ------------------------------------------------------------
combined_mask = trxrd.build_combined_mask(
    sample_image.shape,
    center_xy = (MASK_CENTER_X, MASK_CENTER_Y),
    radius = MASK_RADIUS,
    detector_mask=None,
    mask_path=MASK_FILE,
    plot=False,
    example_image=sample_image,
)
# ------------------------------------------------------------
# Load Data and Compute azimuthal average
# ------------------------------------------------------------
az_result = trxrd.get_azimuthal_average_for_image(
    folder_path=DATA_PATH,
    sample_name=SCAN_NAME,
    filename_scheme=SCAN_TYPE,
    sort_key="image_number",
    delay_sign=DELAY_SIGN,
    poni_path=PONI_FILE,          # handles geometry; set to None and pass centers_xy if using manual geometry
    npt=N_POINTS,
    unit=UNIT,
    nan_radial_range=(NAN_MIN, NAN_MAX),
    integration_mask=combined_mask,
    max_workers=MAX_PROCESSORS,
    progress_interval=100,
    plot=False,                    # plots counts vs image index
)

q         = az_result["radial"]
profiles  = az_result["profiles"]
delays    = az_result["delay"]
counts    = az_result["counts"]
fluences  = az_result["fluence"]

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
# Apply q drift correction
# ------------------------------------------------------------
# results = trxrd.track_peaks(
#     q, profiles.T,
#     ref="mean",          # average pattern is more robust than image 0
#     prominence=150,      # <-- tune this; main knob for detection
#     distance=1,          # min samples between peaks
# )

# s = trxrd.scale_drift_factor(results)
# # Step 2: apply correction
# q_corr, I_corr = trxrd.apply_scale_correction(q, profiles.T, s, kind="cubic")

# # Step 3: verify by re-running peak tracking on corrected data
# results_corr, summary = trxrd.verify_correction(
#     q_corr, I_corr, results,
#     track_peaks_fn=trxrd.track_peaks,
#     prominence=150,    # same as your original call
# )
# profiles_norm = I_corr.T
# q = q_corr

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

    input_file = Path(az_result["file_names"][i])
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