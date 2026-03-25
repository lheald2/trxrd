# Data Processing Algorithm for TRXRD
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import trxrd

# ============================================================
# Load file information
# ============================================================
folder = Path(r"C:\Users\lheald\Box\TRXRDPython\testdata\detimages")
file_names = sorted([f for f in folder.iterdir() if f.suffix.lower() == ".tif"])

print(f"{len(file_names)} TIFF files found in {folder}.")

data_dict = trxrd.get_image_details(
    file_names,
    sort=True,
    filter_data=False,
    plot=True,
)

# ============================================================
# Optional background subtraction
# ============================================================
# background_folder = Path(r"C:\Users\lheald\Box\TRXRDPython\testdata\backgrounds")
#
# bg_result = trxrd.load_background_from_folder(
#     folder=background_folder,
#     average=True,
#     plot=True,
# )
#
# bg_sub_result = trxrd.subtract_background(
#     data_array=data_dict["images"],
#     background=bg_result["background"],
#     plot=True,
# )
#
# images_use = bg_sub_result["corrected_data"]
# data_dict["background_mean"] = bg_result["background_mean"]

# If not using background subtraction:
images_use = data_dict["images"]

# ============================================================
# Apply mask
# ============================================================
mask_file = r"C:\Users\lheald\Box\TRXRDPython\testdata\mask_2021_dec.tif"

masked_data = trxrd.apply_nan_mask(
    images_use,
    mask_path=mask_file,
    plot=True,
)

masked_dict = {
    "images": masked_data,
    "fluence": data_dict["fluence"],
    "delay": data_dict["delay"],
    "image_number": data_dict["image_number"],
    "counts": data_dict["counts"],
}

del data_dict
images_use = masked_dict["images"]

# ============================================================
# Find image centers
# ============================================================
center_dict = trxrd.find_centers_in_stack_radial_parallel(
    images_use,
    center_guess=(1621.0, 746.25),
    search_radius=10,
    mask=None,
    r_min=0,
    r_max=None,
    downsample=1,
    intensity_threshold=None,
    top_percentile=60,
    n_jobs=-1,
    backend="loky",
    verbose=10,
    plot_example=True,
    example_index=0,
    plot_center_vs_image=True,
    image_numbers=masked_dict["image_number"],
)

center_x = np.array(center_dict["center_x"])
center_y = np.array(center_dict["center_y"])
centers = np.column_stack((center_x, center_y))

# ============================================================
# Azimuthal integration
# ============================================================
az_result = trxrd.azimuthal_average_pyfai(
    images=images_use,
    centers=centers,
    use_average_center=False,
    npt=1500,
    unit="q_A^-1",
)

q = az_result["radial"]
profiles = az_result["profiles"]

print("q shape:", q.shape)
print("profiles shape:", profiles.shape)
print("delay shape:", masked_dict["delay"].shape)

# ============================================================
# Normalize per-image profiles
# ============================================================
norm_result = trxrd.normalize_profiles_to_range(
    radial=q,
    profiles=profiles,
    norm_range=(0.9, 1.2),
    mode="mean",
)

norm_profiles = norm_result["normalized_profiles"]

# ============================================================
# Build reference from per-image normalized profiles
# ============================================================
reference_result = trxrd.make_reference_profile(
    profiles=norm_profiles,
    delays=masked_dict["delay"],
)

reference_profile = reference_result["reference_profile"]

plt.figure(figsize=(10, 6))
plt.plot(q, reference_profile, label="Reference Profile", color="red")
plt.xlabel("q (Å⁻¹)")
plt.ylabel("Intensity (a.u.)")
plt.title("Reference Profile")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# Compute per-image delta profiles from normalized data
# ============================================================
delta_result = trxrd.compute_delta_profiles(
    profiles=norm_profiles,
    reference_profile=reference_profile,
    mode="subtract",   # use "relative" for dI/I
)

delta_iq = delta_result["delta_profiles"]

print("delta_iq shape:", delta_iq.shape)

# ============================================================
# Average normalized profiles by delay for visualization
# ============================================================
norm_delay_result = trxrd.average_profiles_by_delay(
    profiles=norm_profiles,
    delays=masked_dict["delay"],
)

unique_delays = norm_delay_result["unique_delays"]
mean_profiles = norm_delay_result["mean_profiles"]
std_profiles = norm_delay_result["std_profiles"]

plt.figure(figsize=(7, 5))
for j in range(min(5, len(unique_delays))):
    plt.plot(q, mean_profiles[j], label=f"{unique_delays[j]:.3g} ps")

plt.xlabel(r"Q ($\AA^{-1}$)")
plt.ylabel("Normalized Intensity")
plt.title("Delay-Averaged Normalized Scattering Profiles")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# Average per-image delta profiles by delay for visualization
# ============================================================
delta_delay_result = trxrd.average_profiles_by_delay(
    profiles=delta_iq,
    delays=masked_dict["delay"],
)

delta_unique_delays = delta_delay_result["unique_delays"]
delta_mean_profiles = delta_delay_result["mean_profiles"]
delta_std_profiles = delta_delay_result["std_profiles"]

plt.figure(figsize=(7, 5))
for j in range(min(5, len(delta_unique_delays))):
    plt.plot(q, delta_mean_profiles[j], label=f"{delta_unique_delays[j]:.3g} ps")

plt.xlabel(r"Q ($\AA^{-1}$)")
plt.ylabel(r"$\Delta I(Q)$")
plt.title("Delay-Averaged Difference Scattering")
plt.axhline(0, color="k", lw=0.8)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.pcolormesh(q, delta_unique_delays, delta_mean_profiles, shading="auto", cmap="RdBu_r")
plt.xlabel(r"Q ($\AA^{-1}$)")
plt.ylabel("Delay")
plt.title(r"$\Delta I(Q, t)$")
plt.colorbar(label=r"$\Delta I$")
plt.clim(-150, 150)
plt.ylim(-2e-9, 10e-9)
plt.tight_layout()
plt.show()

# ============================================================
# Compute q-window lineouts from per-image delta profiles,
# then group by delay for correct uncertainty propagation
# ============================================================
lineout_result = trxrd.lineouts_by_delay_from_per_image_profiles(
    radial=q,
    delta_profiles=delta_iq,
    delays=masked_dict["delay"],
    q_ranges=[(1.95, 2.0), (2.0, 2.1)],
    average_mode="mean",
    error_type="sem",   # or "std"
    plot=True,
)

# Optional access to returned arrays
lineout_delays = lineout_result["unique_delays"]
mean_lineouts = lineout_result["mean_lineouts"]
std_lineouts = lineout_result["std_lineouts"]
sem_lineouts = lineout_result["sem_lineouts"]
counts_per_delay = lineout_result["counts_per_delay"]

print("lineout delays:", lineout_delays)
print("mean_lineouts shape:", mean_lineouts.shape)
print("sem_lineouts shape:", sem_lineouts.shape)
print("counts per delay:", counts_per_delay)
