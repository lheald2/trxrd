# Written by ChatGPT-4, edited by LF Heald 
# This script provides functions for processing TR-XRD data
# March 2026

from pathlib import Path
import re
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import concurrent.futures
from functools import partial
from skimage import feature, filters, exposure
from skimage.morphology import binary_dilation, disk
from joblib import Parallel, delayed
import pyFAI
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from globals import *


_FILENAME_PATTERN = re.compile(
    r"^(?P<sample_name>[A-Za-z0-9_]+)-"
    r"(?P<fluence>[-+]?\d*\.?\d+)fs"
    r"hw(?P<delay>[-+]?\d*\.?\d+(?:e[-+]?\d+)?)"
    r"delay(?P<image_number>\d+)\.tif$",
    re.IGNORECASE,
)


def _get_counts(data_array, plot=False):
    """
    Sum each image over axes (1, 2) and return counts per image.

    Parameters
    ----------
    data_array : np.ndarray
        3D array of shape (N, rows, cols).
    plot : bool, optional
        If True, plot counts versus image index.

    Returns
    -------
    np.ndarray
        1D array of counts for each image.
    """
    if len(data_array) == 0:
        raise ValueError("Input data_array is empty.")

    if data_array.ndim != 3:
        raise ValueError("Input data_array is not 3 dimensional.")

    counts = np.nansum(data_array, axis=(1, 2))

    if plot:
        plt.figure(figsize=FIGSIZE)
        plt.plot(np.arange(len(counts)), counts)
        plt.xlabel("Image index")
        plt.ylabel("Counts")
        plt.title("Counts per image")
        plt.tight_layout()
        plt.show()

    return counts


def _parse_filename(file_name):
    """
    Parse one filename and extract sample_name, fluence, delay, and image number.

    Expected format:
        {sample_name}-{fluence}fshw{delay}delay{image_number}.tif
    Example:
        550nm_re-10.0fshw-1.3e-06delay00036.tif

    Parameters
    ----------
    file_name : str or Path

    Returns
    -------
    tuple
        (sample_name, fluence, delay, image_number)
    """
    name = Path(file_name).name
    match = _FILENAME_PATTERN.search(name)

    if match is None:
        raise ValueError(f"Could not parse filename: {name}")

    sample_name = match.group("sample_name")
    fluence = float(match.group("fluence"))
    delay = float(match.group("delay"))
    delay = -delay  # Positive delay means pump arrives before probe
    image_number = int(match.group("image_number"))

    return sample_name, fluence, delay, image_number


def get_image_details(
    folder_path,
    sample_name=None,
    sort=True,
    filter_data=False,
    plot=False,
):
    """
    Read TIFF images from a folder and extract filename metadata using regex.

    Parameters
    ----------
    folder_path : str or Path
        Folder containing TIFF files.
    sample_name : str or None, optional
        If provided, only keep files whose parsed sample_name matches this
        value (case-insensitive), e.g. "550nm_re".
    sort : bool, optional
        If True, sort data by image_number.
    filter_data : bool or list-like, optional
        If False, use all data.
        If list-like [min_index, max_index], keep only that slice after sorting.
    plot : bool, optional
        If True, show diagnostic plots for the first image and counts.

    Returns
    -------
    dict
        Dictionary containing:
        - "images"        : np.ndarray  (N, rows, cols)
        - "sample_name"   : np.ndarray  (N,)
        - "fluence"       : np.ndarray  (N,)
        - "delay"         : np.ndarray  (N,)
        - "image_number"  : np.ndarray  (N,)
        - "counts"        : np.ndarray  (N,)
        - "file_names"    : np.ndarray  (N,)
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")

    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder}")

    file_names = sorted(folder.glob("*.tif"))
    print(f"{len(file_names)} TIFF files found in {folder}.")

    if len(file_names) == 0:
        raise ValueError(f"No .tif files found in folder: {folder}")

    sample_names = []
    fluence = []
    delay = []
    image_number = []
    cleaned_files = []

    for i, file_name in enumerate(file_names):
        try:
            s_val, f_val, d_val, i_val = _parse_filename(file_name)
        except ValueError:
            # Skip files that do not match expected naming convention
            continue

        if sample_name is not None and s_val.lower() != sample_name.lower():
            continue


        sample_names.append(s_val)
        fluence.append(f_val)
        delay.append(d_val)
        image_number.append(i_val)
        cleaned_files.append(str(file_name))

    if len(cleaned_files) == 0:
        if sample_name is None:
            raise ValueError(
                "No TIFF files in the folder matched the expected filename pattern."
            )
        else:
            raise ValueError(
                f"No TIFF files found for sample_name='{sample_name}' "
                f"that matched the expected filename pattern."
            )

    sample_names = np.array(sample_names, dtype=str)
    fluence = np.array(fluence, dtype=float)
    delay = np.array(delay, dtype=float)
    image_number = np.array(image_number, dtype=int)
    cleaned_files = np.array(cleaned_files, dtype=str)

    if sort:
        idx_sort = np.argsort(image_number)
        sample_names = sample_names[idx_sort]
        fluence = fluence[idx_sort]
        delay = delay[idx_sort]
        image_number = image_number[idx_sort]
        cleaned_files = cleaned_files[idx_sort]

    if isinstance(filter_data, (list, tuple, np.ndarray)):
        if len(filter_data) != 2:
            raise ValueError("filter_data must be False or [min_index, max_index].")

        min_val, max_val = filter_data

        if min_val < 0 or max_val > len(cleaned_files):
            raise ValueError("filter_data range is out of bounds.")

        sample_names = sample_names[min_val:max_val]
        cleaned_files = cleaned_files[min_val:max_val]
        fluence = fluence[min_val:max_val]
        delay = delay[min_val:max_val]
        image_number = image_number[min_val:max_val]

    data_array = tf.imread(list(cleaned_files))
    counts = _get_counts(data_array)

    if plot:
        test = data_array[0]

        plt.figure(figsize=FIGSIZE)

        plt.subplot(1, 3, 1)
        plt.imshow(test, cmap="jet")
        plt.xlabel("Pixel")
        plt.ylabel("Pixel")
        plt.title("Linear Scale")

        plt.subplot(1, 3, 2)
        plt.imshow(np.log(test + 1), cmap="jet")
        plt.xlabel("Pixel")
        plt.ylabel("Pixel")
        plt.title("Log Scale")

        plt.subplot(1, 3, 3)
        plt.hist(test.reshape(-1), bins=30, edgecolor="r", histtype="bar", alpha=0.5)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Pixel Number")
        plt.title("Histogram")
        plt.yscale("log")

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=FIGSIZE)
        plt.plot(counts, "o-")
        plt.xlabel("Image Number")
        plt.ylabel("Counts")
        plt.title("Total Counts")
        plt.tight_layout()
        plt.show()

    data_dict = {
        "images": data_array,
        "sample_name": sample_names,
        "fluence": fluence,
        "delay": delay,
        "image_number": image_number,
        "counts": counts,
        "file_names": cleaned_files,
    }

    return data_dict


def remove_counts(data_dict, std_factor=STD_FACTOR, added_range=None, plot=False):
    """
    Filter images by removing any data where the total counts fall outside
    a threshold defined by std_factor standard deviations from the mean.

    Parameters
    ----------
    data_dict : dict
        Dictionary returned by get_image_details containing:
        - "images"        : np.ndarray  (N, rows, cols)
        - "sample_name"   : np.ndarray  (N,)
        - "fluence"       : np.ndarray  (N,)
        - "delay"         : np.ndarray  (N,)
        - "image_number"  : np.ndarray  (N,)
        - "counts"        : np.ndarray  (N,)
        - "file_names"    : np.ndarray  (N,)
    std_factor : float, optional
        Number of standard deviations from the mean to use as the cutoff.
        Default is STD_FACTOR.
    added_range : list of [min, max] pairs, optional
        Additional index ranges to remove after the std filter.
        Example: [[10, 20], [50, 55]] removes indices 10–19 and 50–54.
    plot : bool, optional
        If True, show diagnostic plot of filtered counts with thresholds.

    Returns
    -------
    dict
        New dictionary with the same keys as the input but with bad images removed.
    """
    if added_range is None:
        added_range = []

    counts = data_dict["counts"]
    init_length = len(counts)

    # --- Standard deviation filter ---
    counts_mean = np.mean(counts)
    counts_std = np.std(counts)

    tc_good = np.squeeze(
        np.where(np.abs(counts - counts_mean) < std_factor * counts_std)
    )

    # Apply std filter to every array in the dictionary
    filtered = {}
    for key, val in data_dict.items():
        if isinstance(val, np.ndarray):
            filtered[key] = val[tc_good]
        else:
            filtered[key] = val

    # --- Additional manual range removal ---
    for rng in added_range:
        for key, val in filtered.items():
            if isinstance(val, np.ndarray):
                filtered[key] = np.concatenate(
                    (val[:rng[0]], val[rng[1]:])
                )

    n_removed = init_length - len(filtered["counts"])
    print(f"{n_removed} images removed from {init_length} initial images")

    # --- Recalculate counts after filtering ---
    filtered["counts"] = _get_counts(filtered["images"])

    if plot:
        plt.figure(figsize=FIGSIZE)

        plt.plot(filtered["counts"], "-d", label="Filtered counts")
        plt.axhline(
            y=counts_mean, color="k", linestyle="-",
            linewidth=1, label="Mean counts",
        )
        plt.axhline(
            y=counts_mean - std_factor * counts_std, color="r",
            linestyle="-", linewidth=0.5, label="Min threshold",
        )
        plt.axhline(
            y=counts_mean + std_factor * counts_std, color="r",
            linestyle="-", linewidth=0.5, label="Max threshold",
        )
        plt.xlabel("Image index")
        plt.ylabel("Counts")
        plt.legend()
        plt.title("Total Counts After Filtering")
        plt.tight_layout()
        plt.show()

    return filtered


def load_background(
    path,
    average=True,
    sort=True,
    plot=False,
    figsize=FIGSIZE,
):
    """
    Load background TIFF image(s) from either a single file or a folder.

    Parameters
    ----------
    path : str or Path
        Path to either:
        - a single .tif / .tiff background image
        - a folder containing multiple .tif / .tiff background images
    average : bool, optional
        If True, return the average background image.
        If False, return the full stack.
    sort : bool, optional
        If True and `path` is a folder, sort files by name.
    plot : bool, optional
        If True, plot a diagnostic figure showing:
        - first background image
        - mean background image
        - histogram of mean background values
    figsize : tuple, optional
        Figure size for diagnostic plot.

    Returns
    -------
    result : dict
        Dictionary containing:
        - "files": list of file paths used
        - "background_stack": np.ndarray of shape (n_bg, rows, cols)
        - "background_mean": np.ndarray of shape (rows, cols)
        - "background": np.ndarray
            Either the mean image if average=True,
            or the full stack if average=False
    """
    path = Path(path)

    if not path.exists():
        raise ValueError(f"Background path does not exist: {path}")

    valid_suffixes = {".tif", ".tiff"}

    # ------------------------------------------------------------
    # Case 1: single background image
    # ------------------------------------------------------------
    if path.is_file():
        if path.suffix.lower() not in valid_suffixes:
            raise ValueError(
                f"Background file must be .tif or .tiff, got: {path.suffix}"
            )

        files = [path]
        background_stack = tf.imread(str(path)).astype(float)

        if background_stack.ndim != 2:
            raise ValueError(
                f"Single background file must load as a 2D image, got shape {background_stack.shape}"
            )

        background_stack = background_stack[None, :, :]

    # ------------------------------------------------------------
    # Case 2: folder of background images
    # ------------------------------------------------------------
    elif path.is_dir():
        files = [
            f for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in valid_suffixes
        ]

        if sort:
            files = sorted(files)

        if len(files) == 0:
            raise ValueError(
                f"No .tif or .tiff background files found in folder: {path}"
            )

        background_stack = tf.imread([str(f) for f in files]).astype(float)

        if background_stack.ndim == 2:
            background_stack = background_stack[None, :, :]

        if background_stack.ndim != 3:
            raise ValueError(
                f"Loaded background data must be 3D after stacking, got shape {background_stack.shape}"
            )

    else:
        raise ValueError(f"Path is neither a file nor a folder: {path}")

    # ------------------------------------------------------------
    # Compute outputs
    # ------------------------------------------------------------
    background_mean = np.nanmean(background_stack, axis=0)
    background_std = np.nanstd(background_stack, axis=0)


    # ------------------------------------------------------------
    # Plot diagnostics
    # ------------------------------------------------------------
    if plot:
        first_image = background_stack[0]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        im0 = axes[0].imshow(first_image, cmap="jet")
        axes[0].set_title("First Background Image")
        axes[0].set_xlabel("Pixel")
        axes[0].set_ylabel("Pixel")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(background_mean, cmap="jet")
        axes[1].set_title("Mean Background Image")
        axes[1].set_xlabel("Pixel")
        axes[1].set_ylabel("Pixel")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].hist(background_mean.ravel(), bins=50, edgecolor="r", alpha=0.5)
        axes[2].set_title("Mean Background Histogram")
        axes[2].set_xlabel("Intensity")
        axes[2].set_ylabel("Pixel count")
        axes[2].set_yscale("log")

        plt.tight_layout()
        plt.show()

    return {
        "files": files,
        "background_stack": background_stack,
        "background_mean": background_mean,
        "background_std": background_std,
    }


def compute_background_azimuthal_average(
    background_input,
    centers=None,
    center_guess=None,
    compute_center_if_missing=True,
    center_from="mean",
    search_radius=20,
    mask=None,
    r_min=0,
    r_max=None,
    downsample=1,
    intensity_threshold=None,
    top_percentile=60,
    npt=5000,
    radial_range=None,
    azimuth_range=None,
    unit="q_A^-1",
    method="bbox",
    correctSolidAngle=True,
    polarization_factor=None,
    dark=None,
    flat=None,
    plot=False,
    image_index=0,
    figsize=FIGSIZE,
    return_dict=True,
):
    """
    Compute azimuthal average(s) for background image data and optionally
    determine the diffraction center from the background image(s).

    Parameters
    ----------
    background_input : dict or np.ndarray
        Either:
        - output dictionary from load_background(...)
        - a 2D background image of shape (rows, cols)
        - a 3D background stack of shape (n_bg, rows, cols)
    centers : tuple or np.ndarray or None, optional
        Diffraction center(s) given as (center_x, center_y).
        May be:
        - None: compute center(s) from background data if allowed
        - a single tuple applied to all background images
        - an array of shape (n_bg, 2)
    center_guess : tuple or None, optional
        Initial guess for center finding in the form (center_y, center_x),
        matching your existing center-finding function convention.
    compute_center_if_missing : bool, optional
        If True and centers is None, compute centers from background data.
    center_from : {"mean", "each"}, optional
        How to compute centers when centers is None:
        - "mean": compute one center from the mean background image and use it for all
        - "each": compute one center per background image
    search_radius, mask, r_min, r_max, downsample, intensity_threshold, top_percentile
        Parameters passed to your center-finding functions.
    npt : int, optional
        Number of radial bins for azimuthal integration.
    radial_range : tuple or None, optional
        Radial range passed to pyFAI.
    azimuth_range : tuple or None, optional
        Azimuthal range passed to pyFAI.
    unit : str, optional
        Radial unit for pyFAI.
    method : str, optional
        Integration method passed to pyFAI.
    correctSolidAngle : bool, optional
        Passed to pyFAI.
    polarization_factor : float or None, optional
        Passed to pyFAI.
    dark : np.ndarray or None, optional
        Dark image passed to pyFAI.
    flat : np.ndarray or None, optional
        Flat-field image passed to pyFAI.
    plot : bool, optional
        If True, plot one example background image and its azimuthal average.
    image_index : int, optional
        Which image to use for example plotting if background data is a stack.
    figsize : tuple, optional
        Figure size for plotting.
    return_dict : bool, optional
        If True, return a dictionary. If False, return tuple.

    Returns
    -------
    result : dict or tuple
        If return_dict=True:
            {
                "radial": np.ndarray of shape (n_q,),
                "background_profiles": np.ndarray of shape (n_bg, n_q),
                "background_profile_mean": np.ndarray of shape (n_q,),
                "background_profile_std": np.ndarray of shape (n_q,),
                "background_images_used": np.ndarray of shape (n_bg, rows, cols),
                "centers_used": np.ndarray of shape (n_bg, 2),
                "center_result": dict or None,
                "pyfai_result": dict,
            }

        If return_dict=False:
            (radial, background_profiles, background_profile_mean)
    """
    # ------------------------------------------------------------
    # Parse input
    # ------------------------------------------------------------
    if isinstance(background_input, dict):
        if "background_stack" in background_input:
            background_images = np.asarray(background_input["background_stack"], dtype=float)
        elif "background_mean" in background_input:
            background_images = np.asarray(background_input["background_mean"], dtype=float)
        else:
            raise ValueError(
                "background_input dictionary must contain 'background_stack' or 'background_mean'."
            )
    else:
        background_images = np.asarray(background_input, dtype=float)

    if background_images.ndim == 2:
        background_images = background_images[None, :, :]
    elif background_images.ndim != 3:
        raise ValueError("background_input must resolve to a 2D or 3D array.")

    n_bg = background_images.shape[0]

    if not (0 <= image_index < n_bg):
        raise ValueError(f"image_index={image_index} is out of bounds for {n_bg} background image(s).")

    # ------------------------------------------------------------
    # Determine centers
    # ------------------------------------------------------------
    center_result = None

    if centers is not None:
        centers_array = np.asarray(centers, dtype=float)

        if centers_array.ndim == 1:
            if centers_array.shape[0] != 2:
                raise ValueError("A single center must have shape (2,).")
            centers_array = np.tile(centers_array, (n_bg, 1))

        elif centers_array.shape != (n_bg, 2):
            raise ValueError(
                f"For {n_bg} background image(s), centers must have shape (2,) or ({n_bg}, 2)."
            )

    else:
        if not compute_center_if_missing:
            raise ValueError(
                "centers is None and compute_center_if_missing=False. "
                "Please provide centers or allow center computation."
            )

        if center_guess is None:
            raise ValueError(
                "center_guess must be provided when computing centers from background data."
            )

        if center_from == "mean":
            mean_bg = np.nanmean(background_images, axis=0)

            center_result = find_diffraction_center_from_guess_radial_fast(
                image=mean_bg,
                center_guess=center_guess,
                search_radius=search_radius,
                mask=mask,
                r_min=r_min,
                r_max=r_max,
                downsample=downsample,
                intensity_threshold=intensity_threshold,
                top_percentile=top_percentile,
                plot=False,
            )

            center_x = center_result["center_x"]
            center_y = center_result["center_y"]
            centers_array = np.tile(np.array([center_x, center_y], dtype=float), (n_bg, 1))

        elif center_from == "each":
            center_result = find_centers_in_stack_radial_parallel(
                data_array=background_images,
                center_guess=center_guess,
                search_radius=search_radius,
                mask=mask,
                r_min=r_min,
                r_max=r_max,
                downsample=downsample,
                intensity_threshold=intensity_threshold,
                top_percentile=top_percentile,
                progress_interval=10,
                max_workers=MAX_PROCESSORS,
            )

            centers_array = np.column_stack((
                center_result["center_x"],
                center_result["center_y"],
            )).astype(float)

        else:
            raise ValueError("center_from must be 'mean' or 'each'")

    # ------------------------------------------------------------
    # Compute azimuthal averages
    # ------------------------------------------------------------
    pyfai_result = azimuthal_average_pyfai(
        images=background_images,
        centers=centers_array,
        npt=npt,
        mask=mask,
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        unit=unit,
        method=method,
        correctSolidAngle=correctSolidAngle,
        polarization_factor=polarization_factor,
        dark=dark,
        flat=flat,
    )

    radial = pyfai_result["radial"]
    background_profiles = pyfai_result["profiles"]

    background_profile_mean = np.nanmean(background_profiles, axis=0)
    background_profile_std = np.nanstd(background_profiles, axis=0)

    # ------------------------------------------------------------
    # Optional diagnostic plot
    # ------------------------------------------------------------
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        im = axes[0].imshow(background_images[image_index], cmap="jet")
        axes[0].set_title("Background Image")
        axes[0].set_xlabel("Pixel")
        axes[0].set_ylabel("Pixel")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        axes[1].plot(radial, background_profiles[image_index], label="Example Background Profile")
        axes[1].plot(radial, background_profile_mean, label="Mean Background Profile", linewidth=2)
        axes[1].set_title("Azimuthal Average of Background")
        axes[1].set_xlabel(unit)
        axes[1].set_ylabel("Intensity")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    if return_dict:
        return {
            "radial": radial,
            "background_profiles": background_profiles,
            "background_profile_mean": background_profile_mean,
            "background_profile_std": background_profile_std,
            "background_images_used": background_images,
            "centers_used": centers_array,
            "center_result": center_result,
            "pyfai_result": pyfai_result,
        }

    return radial, background_profiles, background_profile_mean


def apply_nan_mask(data_array, mask_path, plot=False, image_index=0, figsize=FIGSIZE):
    """
    Apply a binary mask to image data, replacing masked pixels with NaN.

    Parameters
    ----------
    data_array : np.ndarray
        Input image data, either:
        - 2D: (rows, cols)
        - 3D: (n_images, rows, cols)
    mask_path : str or Path
        Path to a mask file containing 0s and 1s.
        Pixels where mask == 0 are replaced with NaN.
    plot : bool, optional
        If True, plot an example original image and masked image.
    image_index : int, optional
        Which image to plot if `data_array` is 3D.
    figsize : tuple, optional
        Figure size for the example plot.

    Returns
    -------
    masked_data : np.ndarray
        Float copy of input data with masked pixels set to NaN.

    Raises
    ------
    ValueError
        If input dimensions are invalid or mask shape does not match image shape.
    """
    mask = tf.imread(Path(mask_path))
    mask_bool = mask == 0

    if data_array.ndim == 2:
        if data_array.shape != mask_bool.shape:
            raise ValueError(
                f"Mask shape {mask_bool.shape} does not match image shape {data_array.shape}."
            )

        original_image = data_array
        masked_data = data_array.astype(float, copy=True)
        masked_data[mask_bool] = np.nan
        masked_image = masked_data

    elif data_array.ndim == 3:
        if data_array.shape[1:] != mask_bool.shape:
            raise ValueError(
                f"Mask shape {mask_bool.shape} does not match image shape {data_array.shape[1:]}."
            )

        if not (0 <= image_index < data_array.shape[0]):
            raise ValueError(
                f"image_index={image_index} is out of bounds for {data_array.shape[0]} images."
            )

        original_image = data_array[image_index]
        masked_data = data_array.astype(float, copy=True)
        masked_data[:, mask_bool] = np.nan
        masked_image = masked_data[image_index]

    else:
        raise ValueError("data_array must be 2D or 3D.")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        im0 = axes[0].imshow(original_image, cmap="jet")
        axes[0].set_title("Original Image")
        axes[0].set_xlabel("Pixel")
        axes[0].set_ylabel("Pixel")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(masked_image, cmap="jet")
        axes[1].set_title("Masked Image")
        axes[1].set_xlabel("Pixel")
        axes[1].set_ylabel("Pixel")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    return masked_data


def _remove_xrays(image, mean_data, std_data, std_factor=STD_FACTOR):
    """
    Replace hot / x-ray spike pixels in a single image with NaN.

    Parameters
    ----------
    image : np.ndarray
        2D image to clean.
    mean_data : np.ndarray
        2D mean image computed from the full stack.
    std_data : np.ndarray
        2D standard deviation image computed from the full stack.
    std_factor : float, optional
        Threshold factor. Pixels with values >= mean + std_factor * std
        are replaced with NaN.

    Returns
    -------
    clean_image : np.ndarray
        2D float array with hot pixels replaced by NaN.
    amt_removed : int
        Number of replaced pixels in this image.
    """
    image = np.asarray(image, dtype=float)
    mean_data = np.asarray(mean_data, dtype=float)
    std_data = np.asarray(std_data, dtype=float)

    if image.ndim != 2:
        raise ValueError("image must be 2D.")
    if mean_data.shape != image.shape or std_data.shape != image.shape:
        raise ValueError("image, mean_data, and std_data must have the same shape.")

    upper_threshold = mean_data + std_factor * std_data
    mask = np.isfinite(image) & np.isfinite(upper_threshold) & (image >= upper_threshold)

    clean_image = image.copy()
    clean_image[mask] = np.nan
    amt_removed = int(np.sum(mask))

    return clean_image, amt_removed


def remove_xrays(data_array, std_factor=STD_FACTOR, plot=False, return_pct=False, figsize=FIGSIZE):
    """
    Replace hot / x-ray spike pixels in a stack of images with NaN.

    Parameters
    ----------
    data_array : np.ndarray
        3D array of shape (n_images, rows, cols).
    std_factor : float, optional
        Threshold factor. Pixels with values >= mean + std_factor * std
        are replaced with NaN.
    plot : bool, optional
        If True, plot percent pixels removed and an example before/after image.
    return_pct : bool, optional
        If True, also return the percent of removed pixels per image.

    Returns
    -------
    clean_data : np.ndarray
        Cleaned image stack with hot pixels replaced by NaN.
    pct_rmv : np.ndarray, optional
        Percent of removed pixels per image. Only returned if return_pct=True.
    """
    data_array = np.asarray(data_array, dtype=float)

    if data_array.ndim != 3:
        raise ValueError("data_array must be a 3D array of shape (n_images, rows, cols).")

    mean_data = np.nanmean(data_array, axis=0)
    std_data = np.nanstd(data_array, axis=0)

    print("Removing hot pixels from all data")

    clean_data = []
    amt_rmv = []

    for image in data_array:
        clean_image, amt = _remove_xrays(
            image=image,
            mean_data=mean_data,
            std_data=std_data,
            std_factor=std_factor,
        )
        clean_data.append(clean_image)
        amt_rmv.append(amt)

    clean_data = np.stack(clean_data)
    n_pixels = data_array.shape[1] * data_array.shape[2]
    pct_rmv = np.asarray(amt_rmv, dtype=float) / n_pixels * 100.0

    if plot:
        plt.figure(figsize=figsize)
        plt.subplot(1, 3, 1)
        plt.plot(pct_rmv)
        plt.title("Percent Pixels Removed")
        plt.xlabel("Image Number")
        plt.ylabel("Percent")

        plt.subplot(1, 3, 2)
        plt.imshow(data_array[0])
        plt.title("Original Image")

        plt.subplot(1, 3, 3)
        plt.imshow(clean_data[0])
        plt.title("Cleaned Image")
        plt.tight_layout()
        plt.show()

    if return_pct:
        return clean_data, pct_rmv
    return clean_data


def remove_xrays_pool(data_array, std_factor=STD_FACTOR, plot=False, return_pct=False, figsize=FIGSIZE):
    """
    Replace hot / x-ray spike pixels in a stack of images with NaN in parallel.

    Parameters
    ----------
    data_array : np.ndarray
        3D array of shape (n_images, rows, cols).
    std_factor : float, optional
        Threshold factor. Pixels with values >= mean + std_factor * std
        are replaced with NaN.
    plot : bool, optional
        If True, plot percent pixels removed and an example before/after image.
    return_pct : bool, optional
        If True, also return the percent of removed pixels per image.

    Returns
    -------
    clean_data : np.ndarray
        Cleaned image stack with hot pixels replaced by NaN.
    pct_rmv : np.ndarray, optional
        Percent of removed pixels per image. Only returned if return_pct=True.
    """
    data_array = np.asarray(data_array, dtype=float)

    if data_array.ndim != 3:
        raise ValueError("data_array must be a 3D array of shape (n_images, rows, cols).")

    mean_data = np.nanmean(data_array, axis=0)
    std_data = np.nanstd(data_array, axis=0)
    print("Removing hot pixels from all data")

    worker = partial(
        _remove_xrays,
        mean_data=mean_data,
        std_data=std_data,
        std_factor=std_factor,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PROCESSORS) as executor:
        results = list(executor.map(worker, data_array))

    clean_data, amt_rmv = zip(*results)
    clean_data = np.stack(clean_data)

    n_pixels = data_array.shape[1] * data_array.shape[2]
    pct_rmv = np.asarray(amt_rmv, dtype=float) / n_pixels * 100.0

    if plot:
        plt.figure(figsize=figsize)
        plt.subplot(1, 3, 1)
        plt.plot(pct_rmv)
        plt.title("Percent Pixels Removed")
        plt.xlabel("Image Number")
        plt.ylabel("Percent")

        plt.subplot(1, 3, 2)
        plt.imshow(data_array[0])
        plt.title("Original Image")

        plt.subplot(1, 3, 3)
        plt.imshow(clean_data[0])
        plt.title("Cleaned Image")
        plt.tight_layout()
        plt.show()

    if return_pct:
        return clean_data, pct_rmv
    return clean_data


def _prepare_valid_pixel_subset(image, mask=None, intensity_threshold=None, top_percentile=None):
    """
    Prepare a subset of valid pixels for faster radial-profile calculations.

    Parameters
    ----------
    image : np.ndarray
        2D image.
    mask : np.ndarray or None
        Boolean mask where True means invalid pixel.
    intensity_threshold : float or None
        Keep only pixels with intensity >= this value.
    top_percentile : float or None
        Keep only pixels at or above this percentile of valid intensities.

    Returns
    -------
    yy : np.ndarray
        y coordinates of selected valid pixels.
    xx : np.ndarray
        x coordinates of selected valid pixels.
    vals : np.ndarray
        Intensity values of selected valid pixels.
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D.")

    img = image.astype(float, copy=False)

    valid = ~np.isnan(img)
    if mask is not None:
        if mask.shape != img.shape:
            raise ValueError("mask must have same shape as image.")
        valid &= ~mask.astype(bool)

    if not np.any(valid):
        raise ValueError("No valid pixels available.")

    if top_percentile is not None:
        thresh = np.percentile(img[valid], top_percentile)
        valid &= img >= thresh

    if intensity_threshold is not None:
        valid &= img >= intensity_threshold

    if not np.any(valid):
        raise ValueError("No pixels remain after brightness filtering.")

    yy, xx = np.nonzero(valid)
    vals = img[yy, xx]
    return yy.astype(float), xx.astype(float), vals


def _radial_profile_from_subset(yy, xx, vals, center_yx):
    """
    Compute radial average profile from a preselected subset of pixels.

    Parameters
    ----------
    yy, xx : np.ndarray
        Pixel coordinates.
    vals : np.ndarray
        Pixel intensities.
    center_yx : tuple
        Center as (cy, cx).

    Returns
    -------
    r : np.ndarray
        Integer radius values.
    radial_mean : np.ndarray
        Mean intensity at each radius.
    """
    cy, cx = center_yx
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.int32)

    radial_sum = np.bincount(rr, weights=vals)
    radial_count = np.bincount(rr)
    radial_mean = radial_sum / np.maximum(radial_count, 1)

    return np.arange(len(radial_mean)), radial_mean


def _profile_sharpness_score(profile):
    """
    Score a radial profile by gradient energy.
    """
    if len(profile) < 3:
        return -np.inf
    grad = np.gradient(profile)
    return np.sum(grad ** 2)


def find_diffraction_center_from_guess_radial_fast(
    image,
    center_guess,
    search_radius=20,
    mask=None,
    r_min=0,
    r_max=None,
    downsample=1,
    intensity_threshold=None,
    top_percentile=None,
    plot=False,
    figsize=FIGSIZE,
):
    """
    Find diffraction center for one image using radial-profile sharpness.

    Parameters
    ----------
    image : np.ndarray
        2D diffraction image.
    center_guess : tuple
        Initial guess as (cy, cx).
    search_radius : int
        Search radius around guessed center.
    mask : np.ndarray or None
        Boolean mask where True means invalid.
    r_min : int
        Minimum radius for scoring.
    r_max : int or None
        Maximum radius for scoring.
    downsample : int
        Integer downsampling factor. 1 means no downsampling.
    intensity_threshold : float or None
        Keep only pixels with intensity >= this threshold.
    top_percentile : float or None
        Keep only pixels at or above this percentile.
    plot : bool
        If True, make diagnostic plots.
    figsize : tuple
        Figure size.

    Returns
    -------
    results : dict
        Dictionary containing center, score, score map, and radial profile.
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D.")
    if downsample < 1 or int(downsample) != downsample:
        raise ValueError("downsample must be an integer >= 1.")

    downsample = int(downsample)
    img = image.astype(float, copy=False)

    if downsample > 1:
        img_work = img[::downsample, ::downsample]
        mask_work = None if mask is None else mask[::downsample, ::downsample]
        guess_y = center_guess[0] / downsample
        guess_x = center_guess[1] / downsample
        search_radius_work = max(1, int(np.ceil(search_radius / downsample)))
        r_min_work = r_min / downsample
        r_max_work = None if r_max is None else r_max / downsample
    else:
        img_work = img
        mask_work = mask
        guess_y, guess_x = center_guess
        search_radius_work = search_radius
        r_min_work = r_min
        r_max_work = r_max

    yy, xx, vals = _prepare_valid_pixel_subset(
        img_work,
        mask=mask_work,
        intensity_threshold=intensity_threshold,
        top_percentile=top_percentile,
    )

    cy_values = np.arange(
        int(np.round(guess_y)) - search_radius_work,
        int(np.round(guess_y)) + search_radius_work + 1,
    )
    cx_values = np.arange(
        int(np.round(guess_x)) - search_radius_work,
        int(np.round(guess_x)) + search_radius_work + 1,
    )

    score_map = np.full((len(cy_values), len(cx_values)), -np.inf, dtype=float)

    best_score = -np.inf
    best_center = None
    best_r = None
    best_profile = None

    for i, cy in enumerate(cy_values):
        for j, cx in enumerate(cx_values):
            r, profile = _radial_profile_from_subset(yy, xx, vals, (cy, cx))

            if r_max_work is None:
                keep = r >= r_min_work
            else:
                keep = (r >= r_min_work) & (r <= r_max_work)

            profile_use = profile[keep]
            if len(profile_use) < 3:
                continue

            score = _profile_sharpness_score(profile_use)
            score_map[i, j] = score

            if score > best_score:
                best_score = score
                best_center = (cy, cx)
                best_r = r
                best_profile = profile

    if best_center is None:
        raise RuntimeError("Could not determine a valid center.")

    best_cy_work, best_cx_work = best_center

    if downsample > 1:
        best_center_full = (best_cy_work * downsample, best_cx_work * downsample)
        best_r_full = best_r * downsample
        cy_values_full = cy_values * downsample
        cx_values_full = cx_values * downsample
    else:
        best_center_full = best_center
        best_r_full = best_r
        cy_values_full = cy_values
        cx_values_full = cx_values

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        img_plot = np.nan_to_num(img, nan=0.0)

        axes[0].imshow(img_plot, cmap="inferno")
        axes[0].plot(center_guess[1], center_guess[0], "co", label="Guess")
        axes[0].plot(best_center_full[1], best_center_full[0], "r+", ms=12, mew=2, label="Best")
        axes[0].set_title("Image with Center")
        axes[0].legend()

        im = axes[1].imshow(
            score_map,
            origin="lower",
            aspect="auto",
            extent=[cx_values_full[0], cx_values_full[-1], cy_values_full[0], cy_values_full[-1]],
            cmap="viridis",
        )
        axes[1].plot(center_guess[1], center_guess[0], "co", label="Guess")
        axes[1].plot(best_center_full[1], best_center_full[0], "r+", ms=12, mew=2, label="Best")
        axes[1].set_xlabel("cx")
        axes[1].set_ylabel("cy")
        axes[1].set_title("Score Map")
        axes[1].legend()
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].plot(best_r_full, best_profile)
        axes[2].set_xlabel("Radius (pixels)")
        axes[2].set_ylabel("Mean intensity")
        axes[2].set_title("Best Radial Profile")

        if r_min > 0:
            axes[2].axvline(r_min, color="gray", linestyle="--")
        if r_max is not None:
            axes[2].axvline(r_max, color="gray", linestyle="--")

        plt.tight_layout()
        plt.show()

    return {
        "center_yx": best_center_full,
        "center_y": best_center_full[0],
        "center_x": best_center_full[1],
        "score": best_score,
        "score_map": score_map,
        "cy_values": cy_values_full,
        "cx_values": cx_values_full,
        "r": best_r_full,
        "radial_profile": best_profile,
    }


def _center_worker(
    idx,
    image,
    center_guess,
    search_radius,
    mask,
    r_min,
    r_max,
    downsample,
    intensity_threshold,
    top_percentile,
):
    """
    Worker for one image, suitable for parallel execution.
    """
    result = find_diffraction_center_from_guess_radial_fast(
        image=image,
        center_guess=center_guess,
        search_radius=search_radius,
        mask=mask,
        r_min=r_min,
        r_max=r_max,
        downsample=downsample,
        intensity_threshold=intensity_threshold,
        top_percentile=top_percentile,
        plot=False,
    )

    cy, cx = result["center_yx"]
    return {
        "index": idx,
        "center_y": cy,
        "center_x": cx,
        "score": result["score"],
        "full_result": result,
    }


def find_centers_in_stack_radial_parallel(
    data_array,
    center_guess = (CENTER_Y, CENTER_X),
    search_radius=20,
    mask=None,
    r_min=0,
    r_max=None,
    downsample=1,
    intensity_threshold=None,
    top_percentile=None,
    max_workers=MAX_PROCESSORS,
    progress_interval=100,
    plot_example=False,
    example_index=0,
    plot_center_vs_image=False,
    image_numbers=None,
    figsize_example=FIGSIZE,
    figsize_trend=FIGSIZE,
):
    """
    Find diffraction centers for a stack of images in parallel using
    concurrent.futures.ThreadPoolExecutor.

    Parameters
    ----------
    data_array : np.ndarray
        3D array of shape (n_images, rows, cols).
    center_guess : tuple
        Fixed center guess as (cy, cx) used for every image.
    search_radius : int
        Search radius around guessed center.
    mask : np.ndarray or None
        Boolean 2D mask where True means invalid.
    r_min : int
        Minimum radius for scoring.
    r_max : int or None
        Maximum radius for scoring.
    downsample : int
        Integer downsampling factor.
    intensity_threshold : float or None
        Keep pixels above this threshold.
    top_percentile : float or None
        Keep pixels at or above this percentile.
    max_workers : int or None
        Number of threads. None lets the executor choose (typically
        min(32, os.cpu_count() + 4)).
    progress_interval : int
        Print progress every this many completed images.
    plot_example : bool
        If True, plot one example result.
    example_index : int
        Which image to use for the example plot.
    plot_center_vs_image : bool
        If True, plot center_x and center_y versus image number/index.
    image_numbers : np.ndarray or None
        X-axis values for trend plots. If None, uses image indices.
    figsize_example : tuple
        Figure size for example plot.
    figsize_trend : tuple
        Figure size for trend plot.

    Returns
    -------
    results : dict
        Dictionary containing:
        - "center_y"         : np.ndarray
        - "center_x"         : np.ndarray
        - "score"            : np.ndarray
        - "image_index"      : np.ndarray
        - "per_image_results": list of per-image dicts
    """
    if data_array.ndim != 3:
        raise ValueError("data_array must be 3D with shape (n_images, rows, cols).")

    n_images = data_array.shape[0]
    print(f"Finding centers for {n_images} images...")

    # ------------------------------------------------------------------
    # Submit all tasks
    # ------------------------------------------------------------------
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

        future_to_idx = {}
        for idx in range(n_images):
            future = executor.submit(
                _center_worker,
                idx=idx,
                image=data_array[idx],
                center_guess=center_guess,
                search_radius=search_radius,
                mask=mask,
                r_min=r_min,
                r_max=r_max,
                downsample=downsample,
                intensity_threshold=intensity_threshold,
                top_percentile=top_percentile,
            )
            future_to_idx[future] = idx

        # --------------------------------------------------------------
        # Collect results with progress reporting
        # --------------------------------------------------------------
        results_list = [None] * n_images
        completed = 0

        for future in concurrent.futures.as_completed(future_to_idx):
            result = future.result()
            results_list[result["index"]] = result
            completed += 1

            if completed % progress_interval == 0 or completed == n_images:
                print(
                    f"  Completed {completed}/{n_images} "
                    f"({100 * completed / n_images:.1f}%)"
                )

    print("Done finding centers.")

    # ------------------------------------------------------------------
    # Unpack into arrays
    # ------------------------------------------------------------------
    center_y = np.array([d["center_y"] for d in results_list], dtype=float)
    center_x = np.array([d["center_x"] for d in results_list], dtype=float)
    score = np.array([d["score"] for d in results_list], dtype=float)
    image_index = np.arange(n_images)
    per_image_results = [d["full_result"] for d in results_list]

    # ------------------------------------------------------------------
    # Optional plots
    # ------------------------------------------------------------------
    if plot_example:
        if not (0 <= example_index < n_images):
            raise ValueError("example_index is out of bounds.")

        example_result = per_image_results[example_index]
        img = data_array[example_index].astype(float)
        img_plot = np.nan_to_num(img, nan=0.0)

        fig, axes = plt.subplots(1, 3, figsize=figsize_example)

        axes[0].imshow(img_plot, cmap="inferno")
        axes[0].plot(center_guess[1], center_guess[0], "co", label="Fixed guess")
        axes[0].plot(
            center_x[example_index], center_y[example_index],
            "r+", ms=12, mew=2, label="Best",
        )
        axes[0].set_title(f"Example Image {example_index}")
        axes[0].legend()

        im = axes[1].imshow(
            example_result["score_map"],
            origin="lower",
            aspect="auto",
            extent=[
                example_result["cx_values"][0],
                example_result["cx_values"][-1],
                example_result["cy_values"][0],
                example_result["cy_values"][-1],
            ],
            cmap="viridis",
        )
        axes[1].plot(center_guess[1], center_guess[0], "co", label="Fixed guess")
        axes[1].plot(
            center_x[example_index], center_y[example_index],
            "r+", ms=12, mew=2, label="Best",
        )
        axes[1].set_xlabel("cx")
        axes[1].set_ylabel("cy")
        axes[1].set_title("Score Map")
        axes[1].legend()
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].plot(example_result["r"], example_result["radial_profile"])
        axes[2].set_xlabel("Radius (pixels)")
        axes[2].set_ylabel("Mean intensity")
        axes[2].set_title("Best Radial Profile")

        if r_min > 0:
            axes[2].axvline(r_min, color="gray", linestyle="--")
        if r_max is not None:
            axes[2].axvline(r_max, color="gray", linestyle="--")

        plt.tight_layout()
        plt.show()

    if plot_center_vs_image:
        xvals = image_numbers if image_numbers is not None else image_index
        xlabel = "Image number" if image_numbers is not None else "Image index"

        fig, axes = plt.subplots(2, 1, figsize=figsize_trend, sharex=True)

        axes[0].plot(xvals, center_x, "o-")
        axes[0].set_ylabel("Center x (pixels)")
        axes[0].set_title("Center Position vs Image")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(xvals, center_y, "o-", color="tab:orange")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel("Center y (pixels)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return {
        "center_y": center_y,
        "center_x": center_x,
        "score": score,
        "image_index": image_index,
        "per_image_results": per_image_results,
    }


def _poly_design_matrix(x, y, order=2):
    """
    Build a 2D polynomial design matrix for smooth background fitting.

    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays.
    order : int, optional
        Polynomial order. Supported values are 0, 1, and 2.

    Returns
    -------
    np.ndarray
        Design matrix of shape (n_points, n_terms).
    """
    if order == 0:
        return np.column_stack([np.ones_like(x)])
    if order == 1:
        return np.column_stack([
            np.ones_like(x),
            x,
            y,
        ])
    if order == 2:
        return np.column_stack([
            np.ones_like(x),
            x,
            y,
            x**2,
            x * y,
            y**2,
        ])
    raise ValueError("Only polynomial orders 0, 1, and 2 are currently supported.")


def build_radial_background_mask(
    shape,
    center,
    r_min=None,
    r_max=None,
    radial_percentile=85,
    extra_mask=None,
):
    """
    Build a boolean mask for background-dominated pixels based on distance from
    the diffraction center.

    Parameters
    ----------
    shape : tuple
        Image shape (rows, cols).
    center : tuple
        Diffraction center as (center_x, center_y) in pixel units.
    r_min : float or None, optional
        Minimum radius to include in the mask. If None, it is set from
        radial_percentile.
    r_max : float or None, optional
        Maximum radius to include in the mask. If None, all radii above
        r_min are included.
    radial_percentile : float, optional
        Percentile of the radius map used to define r_min when r_min is None.
    extra_mask : np.ndarray or None, optional
        Additional boolean mask with True for allowed pixels and False for
        excluded pixels.

    Returns
    -------
    np.ndarray
        Boolean mask with True where pixels are used for background fitting.
    """
    if len(shape) != 2:
        raise ValueError("shape must be length 2: (rows, cols)")

    ny, nx = shape
    cx, cy = center

    yy, xx = np.indices(shape)
    rr = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    if r_min is None:
        r_min = np.nanpercentile(rr, radial_percentile)

    mask = rr >= float(r_min)

    if r_max is not None:
        mask &= rr <= float(r_max)

    if extra_mask is not None:
        extra_mask = np.asarray(extra_mask, dtype=bool)
        if extra_mask.shape != shape:
            raise ValueError("extra_mask must have the same shape as the image.")
        mask &= extra_mask

    return mask


def estimate_background_from_radial_mask(
    image,
    center,
    r_min=None,
    r_max=None,
    radial_percentile=85,
    poly_order=2,
    sigma_clip=3.0,
    alpha=1.0,
    max_iter=3,
    extra_mask=None,
    plot=False,
    figsize=FIGSIZE,
    plot_log=True,
    cmap="jet",
):
    """
    Estimate a smooth 2D background using pixels far from the diffraction
    center, where scattering is assumed to be weakest.

    Parameters
    ----------
    image : np.ndarray
        2D detector image.
    center : tuple
        Diffraction center as (center_x, center_y) in pixel units.
    r_min : float or None, optional
        Minimum radius to include in the background-fit mask.
    r_max : float or None, optional
        Maximum radius to include in the background-fit mask.
    radial_percentile : float, optional
        Radius percentile used to define r_min when r_min is None.
    poly_order : int, optional
        Polynomial order for the smooth 2D background fit.
    sigma_clip : float or None, optional
        Sigma clipping threshold applied within the radial mask.
        Set to None to disable clipping.
    max_iter : int, optional
        Maximum number of sigma-clipping iterations.
    extra_mask : np.ndarray or None, optional
        Boolean mask with True for allowed fitting pixels and False for pixels
        to exclude from the fit.
    plot : bool, optional
        If True, show example plots of the original image, fitted background,
        and corrected image.
    figsize : tuple, optional
        Figure size for diagnostic plots.
    plot_log : bool, optional
        If True, plot images on log scale using log(image - min + 1) style.
        This helps visualize weak background structure.
    cmap : str, optional
        Colormap for image plots.

    Returns
    -------
    dict
        Dictionary containing:
        - "background": fitted 2D background
        - "corrected": image - background
        - "radial_mask": initial radial mask
        - "fit_mask": mask after sigma clipping
        - "coefficients": polynomial coefficients
        - "r_min": radius threshold used
        - "r_max": maximum radius used
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be 2D.")

    ny, nx = image.shape
    yy, xx = np.indices(image.shape)

    radial_mask = build_radial_background_mask(
        shape=image.shape,
        center=center,
        r_min=r_min,
        r_max=r_max,
        radial_percentile=radial_percentile,
        extra_mask=extra_mask,
    )

    valid_mask = radial_mask & np.isfinite(image)
    if not np.any(valid_mask):
        raise ValueError("No valid pixels available for radial background estimation.")

    fit_mask = valid_mask.copy()

    if sigma_clip is not None:
        for _ in range(max_iter):
            vals = image[fit_mask]
            center_val = np.nanmedian(vals)
            spread = 1.4826 * np.nanmedian(np.abs(vals - center_val))

            if not np.isfinite(spread) or spread == 0:
                break

            new_fit_mask = valid_mask & (np.abs(image - center_val) <= sigma_clip * spread)

            if np.array_equal(new_fit_mask, fit_mask):
                break

            fit_mask = new_fit_mask

    # Normalize coordinates to improve numerical stability
    x_norm = (xx - nx / 2.0) / max(nx, 1)
    y_norm = (yy - ny / 2.0) / max(ny, 1)

    x_fit = x_norm[fit_mask].ravel()
    y_fit = y_norm[fit_mask].ravel()
    z_fit = image[fit_mask].ravel()

    A_fit = _poly_design_matrix(x_fit, y_fit, order=poly_order)
    coeffs, _, _, _ = np.linalg.lstsq(A_fit, z_fit, rcond=None)

    A_full = _poly_design_matrix(x_norm.ravel(), y_norm.ravel(), order=poly_order)
    background = (A_full @ coeffs).reshape(image.shape)
    corrected = image - alpha * background

    rr = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    r_min_used = np.nanmin(rr[radial_mask]) if np.any(radial_mask) else np.nan

    if plot:
        def _prep_plot_array(arr, use_log):
            arr = np.asarray(arr, dtype=float)
            if not use_log:
                return arr
            finite = np.isfinite(arr)
            if not np.any(finite):
                return arr
            arr_min = np.nanmin(arr[finite])
            return np.log(np.clip(arr - arr_min, a_min=0, a_max=None) + 1.0)

        image_plot = _prep_plot_array(image, plot_log)
        background_plot = _prep_plot_array(background, plot_log)
        corrected_plot = _prep_plot_array(corrected, plot_log)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        im0 = axes[0].imshow(image_plot, cmap=cmap)
        axes[0].set_title("Original Image")
        axes[0].set_xlabel("Pixel")
        axes[0].set_ylabel("Pixel")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(background_plot, cmap=cmap)
        axes[1].set_title("Estimated Background")
        axes[1].set_xlabel("Pixel")
        axes[1].set_ylabel("Pixel")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(corrected_plot, cmap=cmap)
        axes[2].set_title("Background-Subtracted")
        axes[2].set_xlabel("Pixel")
        axes[2].set_ylabel("Pixel")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    return {
        "background": background,
        "corrected": corrected,
        "radial_mask": radial_mask,
        "fit_mask": fit_mask,
        "coefficients": coeffs,
        "r_min": r_min_used,
        "r_max": r_max,
    }


def _background_worker(
    image,
    center,
    r_min=None,
    r_max=None,
    radial_percentile=85,
    poly_order=2,
    sigma_clip=3.0,
    alpha = 1, 
    max_iter=3,
    extra_mask=None,
):
    """
    Worker for parallel background estimation on one image.
    """
    return estimate_background_from_radial_mask(
        image=image,
        center=center,
        r_min=r_min,
        r_max=r_max,
        radial_percentile=radial_percentile,
        poly_order=poly_order,
        sigma_clip=sigma_clip,
        alpha=alpha,
        max_iter=max_iter,
        extra_mask=extra_mask,
        plot=False,
    )


def subtract_background_from_radial_mask(
    data_array,
    centers,
    r_min=None,
    r_max=None,
    radial_percentile=85,
    poly_order=2,
    sigma_clip=3.0,
    alpha = 1, 
    max_iter=3,
    extra_mask=None,
    plot=False,
    image_index=0,
    figsize=FIGSIZE,
    plot_log=True,
    cmap="jet",
    return_backgrounds=True,
    parallel=True,
    max_workers=MAX_PROCESSORS,
    use_processes=False,
    progress_interval=100,
):
    """
    Estimate and subtract a smooth 2D background from each image using a mask
    defined by radial distance from the diffraction center.

    Parameters
    ----------
    data_array : np.ndarray
        2D image or 3D image stack.
    centers : tuple or np.ndarray
        Diffraction center(s) given as (center_x, center_y). For 3D input, this
        may be a single tuple applied to all images or an array of shape
        (n_images, 2).
    r_min : float or None, optional
        Minimum radius to include in the background-fit mask.
    r_max : float or None, optional
        Maximum radius to include in the background-fit mask.
    radial_percentile : float, optional
        Radius percentile used to define r_min when r_min is None.
    poly_order : int, optional
        Polynomial order for the smooth 2D background fit.
    sigma_clip : float or None, optional
        Sigma clipping threshold applied within the radial mask.
        Set to None to disable clipping.
    max_iter : int, optional
        Maximum number of sigma-clipping iterations.
    extra_mask : np.ndarray or None, optional
        Boolean mask with True for allowed fitting pixels and False for pixels
        to exclude from the fit.
    plot : bool, optional
        If True, plot the original image, estimated background, and corrected
        image for one example image.
    image_index : int, optional
        Image index used for plotting when data_array is 3D.
    figsize : tuple, optional
        Figure size for diagnostic plots.
    plot_log : bool, optional
        If True, show plotted images on a log scale.
    cmap : str, optional
        Colormap for plotted images.
    return_backgrounds : bool, optional
        If True, include the fitted background image(s) in the output.
    parallel : bool, optional
        If True, process stack images in parallel.
    max_workers : int or None, optional
        Number of workers for concurrent.futures.
    use_processes : bool, optional
        If True, use ProcessPoolExecutor instead of ThreadPoolExecutor.
    progress_interval : int, optional
        Print progress every `progress_interval` completed images.

    Returns
    -------
    dict
        Dictionary containing:
        - "corrected_data": background-subtracted image or stack
        - "backgrounds": fitted background image or stack (optional)
        - "radial_masks": mask image or stack used for the fits
        - "fit_masks": sigma-clipped mask image or stack used for the fits
        - "centers_used": center or centers used
    """
    data_array = np.asarray(data_array, dtype=float)

    # --------------------------------------------------------------
    # Single image case
    # --------------------------------------------------------------
    if data_array.ndim == 2:
        result = estimate_background_from_radial_mask(
            image=data_array,
            center=centers,
            r_min=r_min,
            r_max=r_max,
            radial_percentile=radial_percentile,
            poly_order=poly_order,
            sigma_clip=sigma_clip,
            alpha=alpha,
            max_iter=max_iter,
            extra_mask=extra_mask,
            plot=plot,
            figsize=figsize,
            plot_log=plot_log,
            cmap=cmap,
        )

        out = {
            "corrected_data": result["corrected"],
            "radial_masks": result["radial_mask"],
            "fit_masks": result["fit_mask"],
            "centers_used": np.asarray(centers, dtype=float),
        }
        if return_backgrounds:
            out["backgrounds"] = result["background"]

        return out

    # --------------------------------------------------------------
    # Stack case
    # --------------------------------------------------------------
    if data_array.ndim != 3:
        raise ValueError("data_array must be 2D or 3D.")

    n_images = data_array.shape[0]
    centers_array = np.asarray(centers, dtype=float)

    if centers_array.ndim == 1:
        if centers_array.shape[0] != 2:
            raise ValueError("A single center must have shape (2,).")
        centers_array = np.tile(centers_array, (n_images, 1))
    elif centers_array.shape != (n_images, 2):
        raise ValueError("For 3D data, centers must have shape (2,) or (n_images, 2).")

    if not (0 <= image_index < n_images):
        raise IndexError(f"image_index must be between 0 and {n_images - 1}.")

    if progress_interval is None or progress_interval <= 0:
        progress_interval = max(1, n_images // 20)

    # --------------------------------------------------------------
    # Serial execution
    # --------------------------------------------------------------
    if not parallel:
        print("Starting radial background subtraction...")

        results = []
        for i, image in enumerate(data_array):
            result = estimate_background_from_radial_mask(
                image=image,
                center=centers_array[i],
                r_min=r_min,
                r_max=r_max,
                radial_percentile=radial_percentile,
                poly_order=poly_order,
                sigma_clip=sigma_clip,
                max_iter=max_iter,
                extra_mask=extra_mask,
                plot=False,
                figsize=figsize,
                plot_log=plot_log,
                cmap=cmap,
            )
            results.append(result)

            completed = i + 1
            if completed % progress_interval == 0 or completed == n_images:
                print(
                    f"  Completed {completed}/{n_images} "
                    f"({100 * completed / n_images:.1f}%)"
                )

        print("Done with radial background subtraction.")

    # --------------------------------------------------------------
    # Parallel execution
    # --------------------------------------------------------------
    else:
        executor_cls = (
            concurrent.futures.ProcessPoolExecutor
            if use_processes
            else concurrent.futures.ThreadPoolExecutor
        )

        print("Starting radial background subtraction...")

        results = [None] * n_images
        completed = 0

        with executor_cls(max_workers=max_workers) as executor:
            future_to_idx = {}

            for idx in range(n_images):
                future = executor.submit(
                    _background_worker,
                    image=data_array[idx],
                    center=centers_array[idx],
                    r_min=r_min,
                    r_max=r_max,
                    radial_percentile=radial_percentile,
                    poly_order=poly_order,
                    sigma_clip=sigma_clip,
                    max_iter=max_iter,
                    extra_mask=extra_mask,
                )
                future_to_idx[future] = idx

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                result = future.result()
                results[idx] = result

                completed += 1
                if completed % progress_interval == 0 or completed == n_images:
                    print(
                        f"  Completed {completed}/{n_images} "
                        f"({100 * completed / n_images:.1f}%)"
                    )

        print("Done with radial background subtraction.")

    # --------------------------------------------------------------
    # Collect outputs
    # --------------------------------------------------------------
    corrected_list = [result["corrected"] for result in results]
    background_list = [result["background"] for result in results]
    radial_mask_list = [result["radial_mask"] for result in results]
    fit_mask_list = [result["fit_mask"] for result in results]

    out = {
        "corrected_data": np.stack(corrected_list),
        "radial_masks": np.stack(radial_mask_list),
        "fit_masks": np.stack(fit_mask_list),
        "centers_used": centers_array,
    }

    if return_backgrounds:
        out["backgrounds"] = np.stack(background_list)

    # --------------------------------------------------------------
    # Optional plotting of one example image
    # --------------------------------------------------------------
    if plot:
        estimate_background_from_radial_mask(
            image=data_array[image_index],
            center=centers_array[image_index],
            r_min=r_min,
            r_max=r_max,
            radial_percentile=radial_percentile,
            poly_order=poly_order,
            sigma_clip=sigma_clip,
            max_iter=max_iter,
            extra_mask=extra_mask,
            plot=True,
            figsize=figsize,
            plot_log=plot_log,
            cmap=cmap,
        )

    return out


# =====================================================================
# Azimuthal integration (parallel)
# =====================================================================
def normalize_centers(centers, n_images, use_average_center=False):
    """
    Normalize center input into an array of shape (n_images, 2).

    Parameters
    ----------
    centers : tuple, list, or ndarray
        Either:
        - one center: (x_center, y_center)
        - one center per image: shape (n_images, 2)
    n_images : int
        Number of images.
    use_average_center : bool, optional
        If True and centers has shape (n_images, 2), replace all centers
        by the average center.

    Returns
    -------
    centers_out : ndarray
        Array of shape (n_images, 2).
    """
    centers = np.asarray(centers, dtype=float)

    if centers.shape == (2,):
        centers_out = np.tile(centers, (n_images, 1))
    elif centers.shape == (n_images, 2):
        if use_average_center:
            avg_center = np.nanmean(centers, axis=0)
            centers_out = np.tile(avg_center, (n_images, 1))
        else:
            centers_out = centers
    else:
        raise ValueError("centers must have shape (2,) or (n_images, 2)")

    return centers_out


def tilt_to_rotations(tilt_angle, tilt_plane_rotation, rot3=0.0):
    """
    Approximate conversion from detector tilt description to
    pyFAI rotation parameters.

    Parameters
    ----------
    tilt_angle : float
        Detector tilt angle in radians.
    tilt_plane_rotation : float
        Angle of the tilt plane in radians.
    rot3 : float, optional
        In-plane detector rotation in radians.

    Returns
    -------
    rot1, rot2, rot3 : float
        pyFAI rotation parameters.
    """
    rot1 = tilt_angle * np.cos(tilt_plane_rotation)
    rot2 = tilt_angle * np.sin(tilt_plane_rotation)
    return rot1, rot2, rot3


def make_azimuthal_integrator(
    center,
    pixel1=PIXEL1,
    pixel2=PIXEL2,
    distance=DISTANCE,
    wavelength=WAVELENGTH,
    tilt_angle=TILT_ANGLE,
    tilt_plane_rotation=TILT_PLANE_ROTATION,
    rot3=ROT3,
):
    """
    Create a pyFAI AzimuthalIntegrator for a given beam center.

    Parameters
    ----------
    center : tuple or array-like
        (x_center, y_center) in pixel coordinates.
    pixel1, pixel2 : float, optional
        Pixel sizes in meters.
    distance : float, optional
        Sample-detector distance in meters.
    wavelength : float, optional
        Beam wavelength in meters.
    tilt_angle : float, optional
        Detector tilt angle in radians.
    tilt_plane_rotation : float, optional
        Tilt plane rotation angle in radians.
    rot3 : float, optional
        In-plane detector rotation in radians.

    Returns
    -------
    ai : AzimuthalIntegrator
        Configured pyFAI integrator.
    """
    x_center, y_center = center

    poni1 = y_center * pixel1
    poni2 = x_center * pixel2

    rot1, rot2, rot3 = tilt_to_rotations(
        tilt_angle=tilt_angle,
        tilt_plane_rotation=tilt_plane_rotation,
        rot3=rot3,
    )

    ai = AzimuthalIntegrator(
        dist=distance,
        poni1=poni1,
        poni2=poni2,
        rot1=rot1,
        rot2=rot2,
        rot3=rot3,
        pixel1=pixel1,
        pixel2=pixel2,
        wavelength=wavelength,
    )

    return ai


def build_pyfai_mask(image, mask=None):
    """
    Build a pyFAI-compatible mask for one image.

    Parameters
    ----------
    image : ndarray
        2D image, may contain NaN values.
    mask : ndarray or None, optional
        Additional boolean mask with shape matching image.
        True means excluded pixel.

    Returns
    -------
    combined_mask : ndarray or None
        Boolean mask where True means excluded pixel.
    clean_image : ndarray
        Copy of image with non-finite pixels replaced by 0.0.
    """
    nan_mask = ~np.isfinite(image)

    if mask is None:
        combined_mask = nan_mask
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != image.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match image shape {image.shape}"
            )
        combined_mask = nan_mask | mask

    clean_image = np.array(image, dtype=float, copy=True)
    clean_image[nan_mask] = 0.0

    if not np.any(combined_mask):
        combined_mask = None

    return combined_mask, clean_image


def _azimuthal_worker(
    idx,
    image,
    center,
    npt,
    unit,
    radial_range,
    azimuth_range,
    mask,
    dark,
    flat,
    polarization_factor,
    method,
    pixel1,
    pixel2,
    distance,
    wavelength,
    tilt_angle,
    tilt_plane_rotation,
    rot3,
    error_mode,
):
    """
    Worker function for azimuthal integration of a single image.

    Parameters
    ----------
    idx : int
        Image index (used for ordering results).
    image : ndarray
        2D diffraction image.
    center : array-like
        (x_center, y_center) for this image.
    All other parameters match azimuthal_average_pyfai.

    Returns
    -------
    dict
        {"index", "radial", "intensity", "success", "error"}
    """
    try:
        ai = make_azimuthal_integrator(
            center=center,
            pixel1=pixel1,
            pixel2=pixel2,
            distance=distance,
            wavelength=wavelength,
            tilt_angle=tilt_angle,
            tilt_plane_rotation=tilt_plane_rotation,
            rot3=rot3,
        )

        image_mask, clean_image = build_pyfai_mask(image, mask=mask)

        radial, intensity = ai.integrate1d(
            clean_image,
            npt=npt,
            unit=unit,
            radial_range=radial_range,
            azimuth_range=azimuth_range,
            mask=image_mask,
            dark=dark,
            flat=flat,
            polarization_factor=polarization_factor,
            method=method,
        )

        return {
            "index": idx,
            "radial": radial,
            "intensity": intensity,
            "success": True,
            "error": None,
        }

    except Exception as exc:
        msg = f"Integration failed for image index {idx}: {exc}"

        if error_mode == "raise":
            raise RuntimeError(msg) from exc
        elif error_mode == "warn":
            print(f"Warning: {msg}")

        return {
            "index": idx,
            "radial": None,
            "intensity": None,
            "success": False,
            "error": msg,
        }


def azimuthal_average_pyfai(
    images,
    centers,
    use_average_center=False,
    npt=5000,
    unit="q_A^-1",
    radial_range=None,
    azimuth_range=None,
    mask=None,
    dark=DARK,
    flat=FLAT,
    polarization_factor=POLARIZATION_FACTOR,
    method=("bbox", "csr", "cython"),
    pixel1=PIXEL1,
    pixel2=PIXEL2,
    distance=DISTANCE,
    wavelength=WAVELENGTH,
    tilt_angle=TILT_ANGLE,
    tilt_plane_rotation=TILT_PLANE_ROTATION,
    rot3=ROT3,
    return_dict=True,
    error_mode="raise",
    max_workers=None,
    progress_interval=100,
):
    """
    Compute azimuthal averages for a stack of diffraction images using pyFAI,
    parallelized with concurrent.futures.ThreadPoolExecutor.

    Parameters
    ----------
    images : ndarray
        3D array of shape (n_images, ny, nx).
    centers : tuple, list, or ndarray
        Either:
        - one center: (x_center, y_center)
        - one center per image: shape (n_images, 2)
    use_average_center : bool, optional
        If True and centers has shape (n_images, 2), use the average center
        for all images.
    npt : int, optional
        Number of radial bins.
    unit : str, optional
        Output radial unit, e.g. "q_A^-1", "2th_deg", "r_mm".
    radial_range : tuple or None, optional
        Radial integration range in the selected unit.
    azimuth_range : tuple or None, optional
        Azimuthal integration range in degrees.
    mask : ndarray or None, optional
        Additional 2D boolean mask where True means exclude pixel.
    dark : ndarray or None, optional
        Dark correction image.
    flat : ndarray or None, optional
        Flat-field correction image.
    polarization_factor : float or None, optional
        Polarization correction factor.
    method : str or tuple, optional
        pyFAI integration method.
    pixel1, pixel2 : float, optional
        Detector pixel sizes in meters.
    distance : float, optional
        Sample-detector distance in meters.
    wavelength : float, optional
        Beam wavelength in meters.
    tilt_angle : float, optional
        Detector tilt angle in radians.
    tilt_plane_rotation : float, optional
        Tilt plane rotation angle in radians.
    rot3 : float, optional
        In-plane detector rotation in radians.
    return_dict : bool, optional
        If True, return results and metadata in a dictionary.
        If False, return (radial, profiles).
    error_mode : {"raise", "warn", "skip"}, optional
        Behavior if integration fails for an image.
    max_workers : int or None, optional
        Number of threads. None lets the executor choose.
    progress_interval : int, optional
        Print progress every this many completed images.

    Returns
    -------
    result : dict or tuple
        If return_dict=True:
            {
                "radial": radial axis,
                "profiles": integrated intensities, shape (n_images, npt),
                "centers_used": centers actually used, shape (n_images, 2),
                "success": boolean success array, shape (n_images,),
                "geometry": dict of geometry values,
                "unit": selected radial unit,
            }

        If return_dict=False:
            (radial, profiles)
    """
    images = np.asarray(images, dtype=float)

    if images.ndim != 3:
        raise ValueError("images must have shape (n_images, ny, nx)")

    n_images = images.shape[0]

    centers_used = normalize_centers(
        centers,
        n_images=n_images,
        use_average_center=use_average_center,
    )

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != images.shape[1:]:
            raise ValueError(
                f"mask shape {mask.shape} does not match "
                f"image shape {images.shape[1:]}"
            )

    rot1_used, rot2_used, rot3_used = tilt_to_rotations(
        tilt_angle=tilt_angle,
        tilt_plane_rotation=tilt_plane_rotation,
        rot3=rot3,
    )

    print(f"Integrating {n_images} images...")

    profiles = np.full((n_images, npt), np.nan, dtype=float)
    success = np.zeros(n_images, dtype=bool)
    radial_out = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

        future_to_idx = {}
        for idx in range(n_images):
            future = executor.submit(
                _azimuthal_worker,
                idx=idx,
                image=images[idx],
                center=centers_used[idx],
                npt=npt,
                unit=unit,
                radial_range=radial_range,
                azimuth_range=azimuth_range,
                mask=mask,
                dark=dark,
                flat=flat,
                polarization_factor=polarization_factor,
                method=method,
                pixel1=pixel1,
                pixel2=pixel2,
                distance=distance,
                wavelength=wavelength,
                tilt_angle=tilt_angle,
                tilt_plane_rotation=tilt_plane_rotation,
                rot3=rot3,
                error_mode=error_mode,
            )
            future_to_idx[future] = idx

        completed = 0

        for future in concurrent.futures.as_completed(future_to_idx):
            result = future.result()
            i = result["index"]

            if result["success"]:
                profiles[i, :] = result["intensity"]
                success[i] = True

                if radial_out is None:
                    radial_out = result["radial"]

            completed += 1

            if completed % progress_interval == 0 or completed == n_images:
                print(
                    f"  Completed {completed}/{n_images} "
                    f"({100 * completed / n_images:.1f}%)"
                )

    print("Done integrating.")

    if radial_out is None:
        radial_out = np.full(npt, np.nan)

    geometry = {
        "pixel1": pixel1,
        "pixel2": pixel2,
        "distance": distance,
        "wavelength": wavelength,
        "tilt_angle": tilt_angle,
        "tilt_plane_rotation": tilt_plane_rotation,
        "rot1": rot1_used,
        "rot2": rot2_used,
        "rot3": rot3_used,
        "polarization_factor": polarization_factor,
    }

    if return_dict:
        return {
            "radial": radial_out,
            "profiles": profiles,
            "centers_used": centers_used,
            "success": success,
            "geometry": geometry,
            "unit": unit,
        }

    return radial_out, profiles


def plot_normalization_window(
    radial,
    profiles,
    norm_range,
    factors=None,
    normalized_profiles=None,
    plot_indices=None,
    show_normalized=False,
    figsize=FIGSIZE,
    alpha=0.8,
):
    """
    Plot 1D profile(s) and highlight the normalization window.

    Parameters
    ----------
    radial : np.ndarray
        1D radial axis of shape (n_q,).
    profiles : np.ndarray
        2D array of shape (n_profiles, n_q).
    norm_range : tuple
        (min_val, max_val) radial range used for normalization.
    factors : np.ndarray or None, optional
        Normalization factors for each profile. Used in plot labels if provided.
    normalized_profiles : np.ndarray or None, optional
        Precomputed normalized profiles with same shape as profiles.
        Required if show_normalized=True.
    plot_indices : None, int, or sequence of int, optional
        Which profiles to plot. If None, plots the first profile.
    show_normalized : bool, optional
        If True, also plot normalized profiles in a second panel.
    figsize : tuple, optional
        Figure size.
    alpha : float, optional
        Line transparency.

    Returns
    -------
    None
    """
    radial = np.asarray(radial, dtype=float)
    profiles = np.asarray(profiles, dtype=float)

    if radial.ndim != 1:
        raise ValueError("radial must be 1D")
    if profiles.ndim != 2:
        raise ValueError("profiles must be 2D with shape (n_profiles, n_q)")
    if profiles.shape[1] != radial.shape[0]:
        raise ValueError("profiles.shape[1] must match len(radial)")

    if norm_range is None or len(norm_range) != 2:
        raise ValueError("norm_range must be a tuple: (min_val, max_val)")

    rmin, rmax = norm_range
    if rmin >= rmax:
        raise ValueError("norm_range must satisfy min_val < max_val")

    if show_normalized:
        if normalized_profiles is None:
            raise ValueError("normalized_profiles must be provided if show_normalized=True")
        normalized_profiles = np.asarray(normalized_profiles, dtype=float)
        if normalized_profiles.shape != profiles.shape:
            raise ValueError("normalized_profiles must have the same shape as profiles")

    n_profiles = profiles.shape[0]

    if plot_indices is None:
        plot_indices = [0]
    elif np.isscalar(plot_indices):
        plot_indices = [int(plot_indices)]
    else:
        plot_indices = [int(i) for i in plot_indices]

    for idx in plot_indices:
        if idx < 0 or idx >= n_profiles:
            raise ValueError(f"plot index {idx} is out of bounds for {n_profiles} profiles")

    if factors is not None:
        factors = np.asarray(factors, dtype=float)
        if factors.shape != (n_profiles,):
            raise ValueError("factors must have shape (n_profiles,)")

    if show_normalized:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
        axes = np.atleast_1d(axes)

        panel_data = [
            (axes[0], profiles, "Original Profiles"),
            (axes[1], normalized_profiles, "Normalized Profiles"),
        ]
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        panel_data = [(ax, profiles, "Profiles")]

    for ax, data, title in panel_data:
        ax.axvspan(rmin, rmax, color="gold", alpha=0.25, label="Normalization window")
        ax.axvline(rmin, color="goldenrod", linestyle="--")
        ax.axvline(rmax, color="goldenrod", linestyle="--")

        for idx in plot_indices:
            if factors is not None and title == "Original Profiles":
                label = f"Profile {idx} (factor={factors[idx]:.3g})"
            else:
                label = f"Profile {idx}"
            ax.plot(radial, data[idx], alpha=alpha, label=label)

        ax.set_xlabel("Radial coordinate")
        ax.set_ylabel("Intensity")
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.show()


def normalize_profiles_to_range(
    radial,
    profiles,
    norm_range,
    mode="mean",
    return_dict=True,
    plot=False,
    plot_indices=None,
    show_normalized_plot=False,
    plot_factors=False,
    print_factor_stats=False,
    figsize=FIGSIZE,
    alpha=0.8,
):
    """
    Normalize 1D azimuthal profiles using intensity within a specified radial range.

    Parameters
    ----------
    radial : np.ndarray
        1D radial axis of shape (n_q,).
    profiles : np.ndarray
        2D array of shape (n_profiles, n_q).
    norm_range : tuple
        (min_val, max_val) range on the radial axis used for normalization.
    mode : {"mean", "sum", "max"}, optional
        Statistic used to compute the normalization factor inside norm_range:
        - "mean": divide by mean intensity in the range
        - "sum": divide by summed intensity in the range
        - "max": divide by max intensity in the range
    return_dict : bool, optional
        If True, return a dictionary. If False, return tuple.
    plot : bool, optional
        If True, plot the normalization window on selected profiles.
    plot_indices : None, int, or sequence of int, optional
        Which profiles to plot. If None, plots the first profile.
    show_normalized_plot : bool, optional
        If True, also show normalized profiles in a second panel.
    plot_factors : bool, optional
        If True, plot normalization factor versus profile index.
    print_factor_stats : bool, optional
        If True, print summary statistics for the normalization factors.
    figsize : tuple, optional
        Figure size for plotting.
    alpha : float, optional
        Line transparency for plotting.

    Returns
    -------
    result : dict or tuple
        If return_dict=True:
            {
                "normalized_profiles": np.ndarray of shape (n_profiles, n_q),
                "normalization_factors": np.ndarray of shape (n_profiles,),
                "normalization_mask": np.ndarray of shape (n_q,),
                "norm_range": tuple,
                "mode": str,
            }

        If return_dict=False:
            (normalized_profiles, normalization_factors)
    """
    radial = np.asarray(radial, dtype=float)
    profiles = np.asarray(profiles, dtype=float)

    if radial.ndim != 1:
        raise ValueError("radial must be 1D")
    if profiles.ndim != 2:
        raise ValueError("profiles must be 2D with shape (n_profiles, n_q)")
    if profiles.shape[1] != radial.shape[0]:
        raise ValueError("profiles.shape[1] must match len(radial)")
    if norm_range is None or len(norm_range) != 2:
        raise ValueError("norm_range must be a tuple: (min_val, max_val)")

    rmin, rmax = norm_range
    if rmin >= rmax:
        raise ValueError("norm_range must satisfy min_val < max_val")

    norm_mask = (radial >= rmin) & (radial <= rmax)
    if not np.any(norm_mask):
        raise ValueError("No radial points fall inside norm_range")

    norm_region = profiles[:, norm_mask]

    if mode == "mean":
        factors = np.nanmean(norm_region, axis=1)
    elif mode == "sum":
        factors = np.nansum(norm_region, axis=1)
    elif mode == "max":
        factors = np.nanmax(norm_region, axis=1)
    else:
        raise ValueError("mode must be one of: 'mean', 'sum', 'max'")

    if np.any(~np.isfinite(factors)):
        raise ValueError("Some normalization factors are not finite")
    if np.any(factors == 0):
        raise ValueError("Some normalization factors are zero")
    if np.any(factors < 0):
        print("Warning: Some normalization factors are negative")

    normalized_profiles = profiles / factors[:, None]

    if print_factor_stats:
        n_negative = np.sum(factors < 0)
        n_positive = np.sum(factors > 0)
        n_zero = np.sum(factors == 0)

        print("Normalization factor statistics:")
        print(f"  mode: {mode}")
        print(f"  norm_range: {norm_range}")
        print(f"  min:   {np.nanmin(factors):.6g}")
        print(f"  max:   {np.nanmax(factors):.6g}")
        print(f"  mean:  {np.nanmean(factors):.6g}")
        print(f"  std:   {np.nanstd(factors):.6g}")
        print(f"  # < 0: {n_negative}")
        print(f"  # > 0: {n_positive}")
        print(f"  # = 0: {n_zero}")

    if plot:
        plot_normalization_window(
            radial=radial,
            profiles=profiles,
            norm_range=norm_range,
            factors=factors,
            normalized_profiles=normalized_profiles if show_normalized_plot else None,
            plot_indices=plot_indices,
            show_normalized=show_normalized_plot,
            figsize=figsize,
            alpha=alpha,
        )

    if plot_factors:
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(factors))
        ax.plot(x, factors, lw=1.5, label="Normalization factor")
        ax.axhline(0, color="k", linestyle="--", alpha=0.6)

        negative_mask = factors < 0
        if np.any(negative_mask):
            ax.scatter(
                x[negative_mask],
                factors[negative_mask],
                s=18,
                label="Negative factors",
                zorder=3,
            )

        ax.set_title("Normalization Factor vs Profile Index")
        ax.set_xlabel("Profile Index")
        ax.set_ylabel("Normalization Factor")
        ax.legend()
        plt.tight_layout()
        plt.show()

    if return_dict:
        return {
            "normalized_profiles": normalized_profiles,
            "normalization_factors": factors,
            "normalization_mask": norm_mask,
            "norm_range": norm_range,
            "mode": mode,
        }

    return normalized_profiles, factors



def average_profiles_by_delay(
    profiles,
    delays,
    unique_delays=None,
    return_dict=True,
):
    """
    Group 1D scattering profiles by delay and compute average profile
    for each delay.

    Parameters
    ----------
    profiles : np.ndarray
        Array of shape (n_images, n_q).
    delays : np.ndarray
        Array of shape (n_images,) with delay for each image.
    unique_delays : np.ndarray or None, optional
        Specific delay values to use. If None, uses sorted unique delays.
    return_dict : bool, optional
        If True, return a dictionary. If False, return tuple.

    Returns
    -------
    result : dict or tuple
        If return_dict=True:
            {
                "unique_delays": np.ndarray of shape (n_delays,),
                "mean_profiles": np.ndarray of shape (n_delays, n_q),
                "std_profiles": np.ndarray of shape (n_delays, n_q),
                "counts_per_delay": np.ndarray of shape (n_delays,),
                "indices_by_delay": dict
            }

        If return_dict=False:
            (unique_delays, mean_profiles, std_profiles, counts_per_delay)
    """
    profiles = np.asarray(profiles, dtype=float)
    delays = np.asarray(delays, dtype=float)

    if profiles.ndim != 2:
        raise ValueError("profiles must have shape (n_images, n_q)")
    if delays.ndim != 1:
        raise ValueError("delays must be 1D")
    if profiles.shape[0] != len(delays):
        raise ValueError("profiles and delays must have same number of images")

    if unique_delays is None:
        unique_delays = np.array(sorted(np.unique(delays)), dtype=float)
    else:
        unique_delays = np.asarray(unique_delays, dtype=float)

    n_delays = len(unique_delays)
    n_q = profiles.shape[1]

    mean_profiles = np.full((n_delays, n_q), np.nan, dtype=float)
    std_profiles = np.full((n_delays, n_q), np.nan, dtype=float)
    counts_per_delay = np.zeros(n_delays, dtype=int)
    indices_by_delay = {}

    for i, delay_val in enumerate(unique_delays):
        idx = np.where(delays == delay_val)[0]
        indices_by_delay[delay_val] = idx
        counts_per_delay[i] = len(idx)

        if len(idx) == 0:
            continue

        group = profiles[idx]
        mean_profiles[i] = np.nanmean(group, axis=0)
        std_profiles[i] = np.nanstd(group, axis=0)

    if return_dict:
        return {
            "unique_delays": unique_delays,
            "mean_profiles": mean_profiles,
            "std_profiles": std_profiles,
            "counts_per_delay": counts_per_delay,
            "indices_by_delay": indices_by_delay,
        }

    return unique_delays, mean_profiles, std_profiles, counts_per_delay


def make_reference_profile(
    profiles,
    delays,
    reference_selector=None,
    return_dict=True,
):
    """
    Build a reference 1D profile from selected images, defaulting to
    all negative-delay images.

    Parameters
    ----------
    profiles : np.ndarray
        Array of shape (n_images, n_q).
    delays : np.ndarray
        Array of shape (n_images,).
    reference_selector : array-like, callable, or None
        If None, use delays < 0.
        If callable, should take delays and return boolean mask.
        If array-like, interpreted as boolean mask of shape (n_images,).
    return_dict : bool, optional
        If True, return dictionary.

    Returns
    -------
    result : dict or tuple
        If return_dict=True:
            {
                "reference_profile": np.ndarray of shape (n_q,),
                "reference_std": np.ndarray of shape (n_q,),
                "reference_mask": np.ndarray of shape (n_images,),
                "n_reference": int
            }

        If return_dict=False:
            (reference_profile, reference_std, reference_mask)
    """
    profiles = np.asarray(profiles, dtype=float)
    delays = np.asarray(delays, dtype=float)

    if profiles.ndim != 2:
        raise ValueError("profiles must have shape (n_images, n_q)")
    if delays.ndim != 1:
        raise ValueError("delays must be 1D")
    if profiles.shape[0] != len(delays):
        raise ValueError("profiles and delays must have same number of images")

    if reference_selector is None:
        reference_mask = delays < 0
    elif callable(reference_selector):
        reference_mask = np.asarray(reference_selector(delays), dtype=bool)
    else:
        reference_mask = np.asarray(reference_selector, dtype=bool)

    if reference_mask.shape != delays.shape:
        raise ValueError("reference_mask must have same shape as delays")

    if not np.any(reference_mask):
        raise ValueError("No reference images selected")

    ref_group = profiles[reference_mask]
    reference_profile = np.nanmean(ref_group, axis=0)
    reference_std = np.nanstd(ref_group, axis=0)

    if return_dict:
        return {
            "reference_profile": reference_profile,
            "reference_std": reference_std,
            "reference_mask": reference_mask,
            "n_reference": int(np.sum(reference_mask)),
        }

    return reference_profile, reference_std, reference_mask


def compute_delta_profiles(
    profiles,
    reference_profile,
    mode="subtract",
    return_dict=True,
):
    """
    Compute difference profiles relative to a reference profile.

    Parameters
    ----------
    profiles : np.ndarray
        Array of shape (..., n_q), e.g. (n_images, n_q) or (n_delays, n_q).
    reference_profile : np.ndarray
        Array of shape (n_q,).
    mode : {"subtract", "relative"}, optional
        "subtract" computes:
            delta = profiles - reference_profile
        "relative" computes:
            delta = (profiles - reference_profile) / reference_profile
    return_dict : bool, optional
        If True, return dictionary.

    Returns
    -------
    result : dict or np.ndarray
        If return_dict=True:
            {"delta_profiles": delta, "mode": mode}
        else:
            delta
    """
    profiles = np.asarray(profiles, dtype=float)
    reference_profile = np.asarray(reference_profile, dtype=float)

    if profiles.shape[-1] != reference_profile.shape[0]:
        raise ValueError("Last dimension of profiles must match reference_profile length")

    if mode == "subtract":
        delta = profiles - reference_profile
    elif mode == "relative":
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = (profiles - reference_profile) / reference_profile
    else:
        raise ValueError("mode must be 'subtract' or 'relative'")

    if return_dict:
        return {
            "delta_profiles": delta,
            "mode": mode,
        }

    return delta


def lineouts_by_delay_from_per_image_profiles(
    radial,
    delta_profiles,
    delays,
    q_ranges,
    average_mode="mean",
    unique_delays=None,
    error_type="sem",
    plot=True,
    figsize=FIGSIZE,
    marker="o",
    linestyle="-",
    linewidth=1.5,
    alpha_fill=0.25,
    return_dict=True,
):
    """
    Compute time lineouts by averaging per-image delta profiles over specified
    q ranges, then grouping those lineouts by delay.

    Parameters
    ----------
    radial : np.ndarray
        1D q axis of shape (n_q,).
    delta_profiles : np.ndarray
        2D array of shape (n_images, n_q) containing per-image dI or dI/I.
    delays : np.ndarray
        1D array of shape (n_images,) containing delay for each image.
    q_ranges : tuple or list of tuple
        One q-range or a list of q-ranges:
        - (qmin, qmax)
        - [(qmin1, qmax1), (qmin2, qmax2), ...]
    average_mode : {"mean", "sum"}, optional
        How to reduce values inside each q window for each image.
    unique_delays : np.ndarray or None, optional
        Specific delay values to use. If None, uses sorted unique delays.
    error_type : {"std", "sem"}, optional
        Type of uncertainty to plot and return.
    plot : bool, optional
        If True, plot lineouts vs delay with shaded error bands.
    figsize : tuple, optional
        Figure size.
    marker : str, optional
        Marker style.
    linestyle : str, optional
        Line style.
    linewidth : float, optional
        Line width.
    alpha_fill : float, optional
        Alpha for shaded error band.
    return_dict : bool, optional
        If True, return a dictionary.

    Returns
    -------
    result : dict or tuple
        If return_dict=True:
            {
                "unique_delays": np.ndarray of shape (n_delays,),
                "q_ranges": list of tuple,
                "per_image_lineouts": np.ndarray of shape (n_ranges, n_images),
                "mean_lineouts": np.ndarray of shape (n_ranges, n_delays),
                "std_lineouts": np.ndarray of shape (n_ranges, n_delays),
                "sem_lineouts": np.ndarray of shape (n_ranges, n_delays),
                "counts_per_delay": np.ndarray of shape (n_delays,),
                "indices_by_delay": dict,
                "error_type": str,
                "average_mode": str,
            }

        If return_dict=False:
            (unique_delays, mean_lineouts, std_lineouts, sem_lineouts)
    """
    radial = np.asarray(radial, dtype=float)
    delta_profiles = np.asarray(delta_profiles, dtype=float)
    delays = np.asarray(delays, dtype=float)

    if radial.ndim != 1:
        raise ValueError("radial must be 1D")
    if delta_profiles.ndim != 2:
        raise ValueError("delta_profiles must have shape (n_images, n_q)")
    if delays.ndim != 1:
        raise ValueError("delays must be 1D")
    if delta_profiles.shape[1] != radial.shape[0]:
        raise ValueError("delta_profiles.shape[1] must match len(radial)")
    if delta_profiles.shape[0] != delays.shape[0]:
        raise ValueError("delta_profiles.shape[0] must match len(delays)")

    if isinstance(q_ranges, tuple) and len(q_ranges) == 2 and np.isscalar(q_ranges[0]):
        q_ranges = [q_ranges]
    else:
        q_ranges = list(q_ranges)

    if unique_delays is None:
        unique_delays = np.array(sorted(np.unique(delays)), dtype=float)
    else:
        unique_delays = np.asarray(unique_delays, dtype=float)

    n_images = delta_profiles.shape[0]
    n_ranges = len(q_ranges)
    n_delays = len(unique_delays)

    per_image_lineouts = np.full((n_ranges, n_images), np.nan, dtype=float)
    q_masks = []

    for i, q_range in enumerate(q_ranges):
        if len(q_range) != 2:
            raise ValueError("Each q range must be a tuple: (qmin, qmax)")

        qmin, qmax = q_range
        if qmin >= qmax:
            raise ValueError(f"Invalid q range {q_range}: must satisfy qmin < qmax")

        q_mask = (radial >= qmin) & (radial <= qmax)
        if not np.any(q_mask):
            raise ValueError(f"No radial points fall inside q range {q_range}")

        q_masks.append(q_mask)
        region = delta_profiles[:, q_mask]

        if average_mode == "mean":
            per_image_lineouts[i] = np.nanmean(region, axis=1)
        elif average_mode == "sum":
            per_image_lineouts[i] = np.nansum(region, axis=1)
        else:
            raise ValueError("average_mode must be one of: 'mean', 'sum'")

    mean_lineouts = np.full((n_ranges, n_delays), np.nan, dtype=float)
    std_lineouts = np.full((n_ranges, n_delays), np.nan, dtype=float)
    sem_lineouts = np.full((n_ranges, n_delays), np.nan, dtype=float)
    counts_per_delay = np.zeros(n_delays, dtype=int)
    indices_by_delay = {}

    for j, delay_val in enumerate(unique_delays):
        idx = np.where(delays == delay_val)[0]
        indices_by_delay[delay_val] = idx
        counts_per_delay[j] = len(idx)

        if len(idx) == 0:
            continue

        group = per_image_lineouts[:, idx]
        mean_lineouts[:, j] = np.nanmean(group, axis=1)
        std_lineouts[:, j] = np.nanstd(group, axis=1)

        if len(idx) > 0:
            sem_lineouts[:, j] = std_lineouts[:, j] / np.sqrt(len(idx))

    if error_type == "std":
        errors = std_lineouts
    elif error_type == "sem":
        errors = sem_lineouts
    else:
        raise ValueError("error_type must be 'std' or 'sem'")

    if plot:
        plt.figure(figsize=figsize)

        for i, q_range in enumerate(q_ranges):
            qmin, qmax = q_range
            label = f"{qmin:.3g} to {qmax:.3g}"

            plt.plot(
                unique_delays,
                mean_lineouts[i],
                marker=marker,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label,
            )

            plt.fill_between(
                unique_delays,
                mean_lineouts[i] - errors[i],
                mean_lineouts[i] + errors[i],
                alpha=alpha_fill,
            )

        plt.xlabel("Delay")
        plt.ylabel("Averaged signal")
        plt.title("Lineouts vs Delay")
        plt.legend(title="q range")
        plt.tight_layout()
        plt.show()

    if return_dict:
        return {
            "unique_delays": unique_delays,
            "q_ranges": q_ranges,
            "q_masks": q_masks,
            "per_image_lineouts": per_image_lineouts,
            "mean_lineouts": mean_lineouts,
            "std_lineouts": std_lineouts,
            "sem_lineouts": sem_lineouts,
            "counts_per_delay": counts_per_delay,
            "indices_by_delay": indices_by_delay,
            "error_type": error_type,
            "average_mode": average_mode,
        }

    return unique_delays, mean_lineouts, std_lineouts, sem_lineouts

