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

Note: This code relies on the globals.py configuration file for paths, parameters, and settings. 
Make sure to update the paths and parameters in globals.py
"""

from pathlib import Path
import numpy as np

import trxrd
from globals import *

# Check number of files in folder 
file_names = sorted(DATA_PATH.glob(f"{SCAN_NAME}*.tif"))
print(f"{len(file_names)} TIFF files found in {DATA_PATH}.")


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
data_dict = trxrd.get_image_details(
    folder_path=DATA_PATH,
    sample_name=SCAN_NAME,   
    sort=True,
    filter_data=False,
    plot=False,
)

print(data_dict.keys())
print("Images shape:", data_dict["images"].shape)
print("Counts shape:", data_dict["counts"].shape)
print("Unique delays:", np.unique(data_dict["delay"]))

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
az_result = trxrd.azimuthal_average_pyfai(
    images=data_dict["images"],
    centers_xy=(CENTER_X, CENTER_Y),   # note: (x, y)
    polarization_factor=POLARIZATION_FACTOR,
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

delays = data_dict["delay"]
# If you also want fluence, include it if available:
fluence = data_dict.get("fluence", None)

for i, profile in enumerate(profiles_norm):

    # Skip bad profiles
    if not np.any(np.isfinite(profile)):
        print(f"Skipping index {i} (all NaN)")
        continue

    # --------------------------------------------------------
    # Build filename
    # --------------------------------------------------------
    delay_val = delays[i]

    # Format delay nicely (adjust as needed)
    delay_str = f"{delay_val:.0f}fs"

    if fluence is not None:
        fluence_val = fluence[i]
        filename = f"{SCAN_NAME}_{fluence_val:.0f}uJ_{delay_str}_{i:05d}.dat"
    else:
        filename = f"{SCAN_NAME}_{delay_str}_{i:05d}.dat"

    output_file = SAVE_PATH / filename

    # --------------------------------------------------------
    # Save (q, I) columns
    # --------------------------------------------------------
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