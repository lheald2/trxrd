# Written by ChatGPT-4, edited by LF Heald 
# This script provides functions for processing TR-XRD data
# March 2026

from pathlib import Path
import re
import numpy as np
import numpy.ma as ma
import tifffile as tf
import matplotlib.pyplot as plt
import concurrent.futures
from functools import partial
from scipy import sparse
from scipy.sparse.linalg import spsolve
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
    Sum image counts for either one image or a stack.

    Parameters
    ----------
    data_array : np.ndarray
        2D image or 3D image stack.
    plot : bool, optional
        If True, plot counts versus image index.

    Returns
    -------
    np.ndarray
        1D array of counts of length n_images.
        For a 2D input image, returns an array of length 1.
    """
    image_stack, _ = _as_image_stack(data_array, name="data_array")

    if image_stack.shape[0] == 0:
        raise ValueError("Input data_array is empty.")

    counts = np.nansum(image_stack, axis=(1, 2))

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

    file_names = sorted(folder.glob(f"{sample_name}*.tif"))
    print(f"{len(file_names)} TIFF files found in {folder} with scan name {sample_name}.")

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


def remove_counts(
    data_dict,
    std_factor=STD_FACTOR,
    added_range=None,
    plot=False,
    return_dict=True,
):
    """
    Remove images whose total counts fall outside a threshold defined by
    `std_factor` standard deviations from the mean.

    This function operates on the dictionary returned by `get_image_details(...)`
    and filters all array-like entries consistently using the same image mask.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing image data and associated metadata. Expected keys:
        - "images"       : np.ndarray of shape (n_images, rows, cols)
        - "counts"       : np.ndarray of shape (n_images,)
        - "sample_name"  : np.ndarray of shape (n_images,)
        - "fluence"      : np.ndarray of shape (n_images,)
        - "delay"        : np.ndarray of shape (n_images,)
        - "image_number" : np.ndarray of shape (n_images,)
        - "file_names"   : np.ndarray of shape (n_images,)
    std_factor : float, optional
        Number of standard deviations from the mean used to define the
        acceptable counts range.
    added_range : list of [min_index, max_index] pairs or None, optional
        Additional index ranges to remove after the counts-based filter.
        Each pair removes entries in Python slice style:
        [min_index, max_index) removes indices min_index through max_index-1.

        Example:
        - [[10, 20], [50, 55]] removes indices 10-19 and 50-54
          after the standard-deviation filter has been applied.
    plot : bool, optional
        If True, plot the filtered counts with the original mean and threshold
        lines for diagnostic purposes.
    return_dict : bool, optional
        If True, return a dictionary.
        If False, return only the filtered data dictionary.

    Returns
    -------
    result : dict or dict
        If return_dict=True:
            {
                "filtered_data": dict,
                "counts_mean_initial": float,
                "counts_std_initial": float,
                "lower_threshold": float,
                "upper_threshold": float,
                "good_mask_initial": np.ndarray of shape (n_images,),
                "n_removed": int,
                "n_initial": int,
                "n_final": int,
                "added_range": list,
            }

        If return_dict=False:
            filtered_data

        `filtered_data` has the same structure as the input data_dict, but with
        bad images removed and counts recalculated from the filtered image stack.

    Raises
    ------
    ValueError
        If required keys are missing, if counts and images are inconsistent,
        or if added_range contains invalid ranges.
    """
    if added_range is None:
        added_range = []

    required_keys = ["images", "counts"]
    for key in required_keys:
        if key not in data_dict:
            raise ValueError(f"data_dict is missing required key: '{key}'")

    images = np.asarray(data_dict["images"], dtype=float)
    counts = np.asarray(data_dict["counts"], dtype=float)

    image_stack, _ = _as_image_stack(images, name="data_dict['images']")
    n_initial = image_stack.shape[0]

    if counts.ndim != 1:
        raise ValueError("data_dict['counts'] must be 1D.")
    if len(counts) != n_initial:
        raise ValueError(
            "Length of data_dict['counts'] must match number of images in data_dict['images']."
        )

    # ------------------------------------------------------------
    # Initial counts-based filtering
    # ------------------------------------------------------------
    counts_mean_initial = np.nanmean(counts)
    counts_std_initial = np.nanstd(counts)

    lower_threshold = counts_mean_initial - std_factor * counts_std_initial
    upper_threshold = counts_mean_initial + std_factor * counts_std_initial

    good_mask_initial = (
        np.isfinite(counts) &
        (counts >= lower_threshold) &
        (counts <= upper_threshold)
    )

    filtered_data = {}
    for key, val in data_dict.items():
        if isinstance(val, np.ndarray):
            if len(val) == n_initial:
                filtered_data[key] = val[good_mask_initial]
            else:
                filtered_data[key] = val
        else:
            filtered_data[key] = val

    # ------------------------------------------------------------
    # Manual range removal after counts filtering
    # ------------------------------------------------------------
    for rng in added_range:
        if len(rng) != 2:
            raise ValueError(
                "Each entry in added_range must be [min_index, max_index]."
            )

        start, stop = rng
        n_current = len(filtered_data["images"])

        if start < 0 or stop < 0 or start > stop or stop > n_current:
            raise ValueError(
                f"Invalid removal range {rng} for current filtered length {n_current}."
            )

        keep_mask = np.ones(n_current, dtype=bool)
        keep_mask[start:stop] = False

        for key, val in filtered_data.items():
            if isinstance(val, np.ndarray) and len(val) == n_current:
                filtered_data[key] = val[keep_mask]

    # ------------------------------------------------------------
    # Recalculate counts after filtering
    # ------------------------------------------------------------
    filtered_data["counts"] = _get_counts(filtered_data["images"])

    n_final = len(filtered_data["counts"])
    n_removed = n_initial - n_final

    print(f"{n_removed} images removed from {n_initial} initial images")

    # ------------------------------------------------------------
    # Plot diagnostics
    # ------------------------------------------------------------
    if plot:
        plt.figure(figsize=FIGSIZE)

        plt.plot(filtered_data["counts"], "-d", label="Filtered counts")
        plt.axhline(
            y=counts_mean_initial,
            color="k",
            linestyle="-",
            linewidth=1,
            label="Initial mean counts",
        )
        plt.axhline(
            y=lower_threshold,
            color="r",
            linestyle="--",
            linewidth=1,
            label="Lower threshold",
        )
        plt.axhline(
            y=upper_threshold,
            color="r",
            linestyle="--",
            linewidth=1,
            label="Upper threshold",
        )

        plt.xlabel("Image index")
        plt.ylabel("Counts")
        plt.title("Total Counts After Filtering")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if return_dict:
        return {
            "filtered_data": filtered_data,
            "counts_mean_initial": counts_mean_initial,
            "counts_std_initial": counts_std_initial,
            "lower_threshold": lower_threshold,
            "upper_threshold": upper_threshold,
            "good_mask_initial": good_mask_initial,
            "n_removed": n_removed,
            "n_initial": n_initial,
            "n_final": n_final,
            "added_range": added_range,
        }

    return filtered_data


def _as_image_stack(images, name="images"):
    """
    Convert a 2D image or 3D image stack into a 3D stack.
    """
    arr = np.asarray(images, dtype=float)

    if arr.ndim == 2:
        return arr[None, :, :], True
    if arr.ndim == 3:
        return arr, False

    raise ValueError(f"{name} must be 2D or 3D, got shape {arr.shape}")


def _restore_image_dimensionality(image_stack, input_was_2d):
    """
    Return a 2D image if the original input was 2D, otherwise return the 3D stack.
    """
    if input_was_2d:
        return image_stack[0]
    return image_stack


def _normalize_centers_xy(centers, n_images, use_average_center=False):
    """
    Normalize center input into an array of shape (n_images, 2) in (x, y) order.
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
        raise ValueError(
            f"centers must have shape (2,) or ({n_images}, 2), got {centers.shape}"
        )

    return centers_out


def load_background(
    path,
    sort=True,
    plot=False,
    figsize=FIGSIZE,
):
    """
    Load background TIFF image(s) from either a single file or a folder.
    """
    path = Path(path)

    if not path.exists():
        raise ValueError(f"Background path does not exist: {path}")

    valid_suffixes = {".tif", ".tiff"}

    if path.is_file():
        if path.suffix.lower() not in valid_suffixes:
            raise ValueError(f"Background file must be .tif or .tiff, got: {path.suffix}")

        files = [path]
        background_stack = tf.imread(str(path)).astype(float)

        if background_stack.ndim != 2:
            raise ValueError(
                f"Single background file must load as a 2D image, got shape {background_stack.shape}"
            )

        background_stack = background_stack[None, :, :]

    elif path.is_dir():
        files = [
            f for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in valid_suffixes
        ]

        if sort:
            files = sorted(files)

        if len(files) == 0:
            raise ValueError(f"No .tif or .tiff background files found in folder: {path}")

        background_stack = tf.imread([str(f) for f in files]).astype(float)

        if background_stack.ndim == 2:
            background_stack = background_stack[None, :, :]

        if background_stack.ndim != 3:
            raise ValueError(
                f"Loaded background data must be 3D after stacking, got shape {background_stack.shape}"
            )

    else:
        raise ValueError(f"Path is neither a file nor a folder: {path}")

    background_mean = np.nanmean(background_stack, axis=0)
    background_std = np.nanstd(background_stack, axis=0)

    if plot:
        first_image = background_stack[0]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        im0 = axes[0].imshow(first_image, cmap="jet")
        axes[0].set_title("First Background Image")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(background_mean, cmap="jet")
        axes[1].set_title("Mean Background Image")
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
        "n_images": background_stack.shape[0],
    }


def subtract_background(
    data_array,
    background,
    average_background=True,
    plot=False,
    image_index=0,
    figsize=(12, 4),
):
    """
    Subtract a background image from diffraction data.

    Parameters
    ----------
    data_array : np.ndarray
        2D image or 3D image stack.
    background : np.ndarray
        2D background image or 3D background stack.
    average_background : bool, optional
        If True and background is 3D, use its mean.
        If False and background is 3D, background must match data shape image-by-image.
    plot : bool, optional
        If True, plot one example.
    image_index : int, optional
        Which image to plot.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    dict
        {
            "corrected_data": same dimensionality as input data_array,
            "background_used": 2D or 3D background actually subtracted,
            "background_mean": 2D mean background image,
        }
    """
    image_stack, input_was_2d = _as_image_stack(data_array, name="data_array")
    bg_stack, bg_input_was_2d = _as_image_stack(background, name="background")

    n_images = image_stack.shape[0]

    if not (0 <= image_index < n_images):
        raise ValueError(
            f"image_index={image_index} is out of bounds for {n_images} image(s)"
        )

    background_mean = np.nanmean(bg_stack, axis=0)

    if bg_input_was_2d:
        background_used = background_mean
        if image_stack.shape[1:] != background_used.shape:
            raise ValueError(
                f"Background shape {background_used.shape} does not match image shape {image_stack.shape[1:]}"
            )
        corrected_stack = image_stack - background_used[None, :, :]

    else:
        if average_background:
            background_used = background_mean
            if image_stack.shape[1:] != background_used.shape:
                raise ValueError(
                    f"Background shape {background_used.shape} does not match image shape {image_stack.shape[1:]}"
                )
            corrected_stack = image_stack - background_used[None, :, :]
        else:
            if bg_stack.shape != image_stack.shape:
                raise ValueError(
                    f"background stack shape {bg_stack.shape} does not match data stack shape {image_stack.shape}"
                )
            background_used = bg_stack
            corrected_stack = image_stack - background_used

    if background_used.ndim == 2:
        background_image = background_used
    else:
        background_image = background_used[image_index]

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        im0 = axes[0].imshow(image_stack[image_index], cmap="jet")
        axes[0].set_title("Original Image")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(background_image, cmap="jet")
        axes[1].set_title("Background Image")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(corrected_stack[image_index], cmap="jet")
        axes[2].set_title("Background-Subtracted Image")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    return {
        "corrected_data": _restore_image_dimensionality(corrected_stack, input_was_2d),
        "background_used": background_used,
        "background_mean": background_mean,
    }


def apply_nan_mask(
    data_array,
    mask_path,
    plot=False,
    image_index=0,
    figsize=FIGSIZE,
    use_shared_color_scale=True,
):
    """
    Apply a binary mask to image data, replacing masked pixels with NaN.

    This function accepts either a single 2D image or a 3D image stack.
    Internally, the input is converted to a stack using `_as_image_stack`,
    the mask is broadcast across all images, and the original dimensionality
    is restored before returning.

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
        Which image to plot if the input is a stack.
        If the input is a single 2D image, image_index must be 0.
    figsize : tuple, optional
        Figure size for the example plot.

    Returns
    -------
    masked_data : np.ndarray
        Float copy of input data with masked pixels set to NaN.
        Returns:
        - 2D array if input was 2D
        - 3D array if input was 3D

    Raises
    ------
    ValueError
        If the input dimensions are invalid, if the mask shape does not match
        the image shape, or if image_index is out of bounds.
    """
    image_stack, input_was_2d = _as_image_stack(data_array, name="data_array")

    mask = tf.imread(Path(mask_path))
    mask_bool = np.asarray(mask == 0, dtype=bool)

    if image_stack.shape[1:] != mask_bool.shape:
        raise ValueError(
            f"Mask shape {mask_bool.shape} does not match image shape {image_stack.shape[1:]}."
        )

    n_images = image_stack.shape[0]
    if not (0 <= image_index < n_images):
        raise ValueError(
            f"image_index={image_index} is out of bounds for {n_images} image(s)."
        )

    masked_stack = image_stack.astype(float, copy=True)

    # Broadcast the 2D mask across the full image stack without altering
    # unmasked pixel values.
    masked_stack = np.where(mask_bool[None, :, :], np.nan, masked_stack)

    if plot:
        original_image = image_stack[image_index]
        masked_image = masked_stack[image_index]

        if use_shared_color_scale:
            finite_vals = original_image[np.isfinite(original_image)]
            vmin = np.nanmin(finite_vals)
            vmax = np.nanmax(finite_vals)
        else:
            vmin = None
            vmax = None

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        im0 = axes[0].imshow(original_image, cmap="jet", vmin=vmin, vmax=vmax)
        axes[0].set_title("Original Image")
        axes[0].set_xlabel("Pixel")
        axes[0].set_ylabel("Pixel")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(masked_image, cmap="jet", vmin=vmin, vmax=vmax)
        axes[1].set_title("Masked Image")
        axes[1].set_xlabel("Pixel")
        axes[1].set_ylabel("Pixel")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    return _restore_image_dimensionality(masked_stack, input_was_2d)


def _remove_xrays(
    image,
    mean_image,
    std_image,
    std_factor=STD_FACTOR,
):
    """
    Replace hot pixels in a single image with NaN using a threshold based on
    the stack mean and standard deviation.

    Parameters
    ----------
    image : np.ndarray
        2D image to clean.
    mean_image : np.ndarray
        2D mean image computed from the full stack.
    std_image : np.ndarray
        2D standard deviation image computed from the full stack.
    std_factor : float, optional
        Threshold multiplier. Pixels satisfying

            image >= mean_image + std_factor * std_image

        are replaced with NaN.

    Returns
    -------
    result : dict
        Dictionary containing:
        - "clean_image" : np.ndarray
            2D cleaned image with hot pixels replaced by NaN.
        - "bad_mask" : np.ndarray
            Boolean mask of removed pixels.
        - "n_removed" : int
            Number of removed pixels.
        - "pct_removed" : float
            Percent of removed pixels in the image.
    """
    image = np.asarray(image, dtype=float)
    mean_image = np.asarray(mean_image, dtype=float)
    std_image = np.asarray(std_image, dtype=float)

    if image.ndim != 2:
        raise ValueError("image must be 2D.")
    if mean_image.shape != image.shape:
        raise ValueError("mean_image must have the same shape as image.")
    if std_image.shape != image.shape:
        raise ValueError("std_image must have the same shape as image.")

    upper_threshold = mean_image + std_factor * std_image
    bad_mask = image >= upper_threshold

    clean_image = image.copy()
    clean_image[bad_mask] = np.nan

    n_removed = int(np.sum(bad_mask))
    pct_removed = 100.0 * n_removed / image.size

    return {
        "clean_image": clean_image,
        "bad_mask": bad_mask,
        "n_removed": n_removed,
        "pct_removed": pct_removed,
    }


def remove_xrays(
    data_array,
    std_factor=STD_FACTOR,
    plot=False,
    image_index=0,
    return_dict=True,
):
    """
    Remove hot pixels from one image or a stack of images using a threshold
    based on the stack mean and standard deviation.

    This function converts the input to a 3D image stack internally, computes
    the stack mean and standard deviation, removes hot pixels from each image,
    and then restores the original dimensionality before returning.

    Parameters
    ----------
    data_array : np.ndarray
        Input image data, either:
        - 2D: (rows, cols)
        - 3D: (n_images, rows, cols)
    std_factor : float, optional
        Threshold multiplier used to identify hot pixels.
    plot : bool, optional
        If True, plot:
        - percent of removed pixels vs image index
        - one example original image
        - one example cleaned image
    image_index : int, optional
        Image index used for plotting when the input is a stack.
    return_dict : bool, optional
        If True, return a dictionary.
        If False, return only the cleaned data array.

    Returns
    -------
    result : dict or np.ndarray
        If return_dict=True:
            {
                "clean_data": np.ndarray,
                "pct_removed": np.ndarray of shape (n_images,),
                "n_removed": np.ndarray of shape (n_images,),
                "mean_image": np.ndarray of shape (rows, cols),
                "std_image": np.ndarray of shape (rows, cols),
                "input_was_2d": bool,
            }

        If return_dict=False:
            clean_data

        `clean_data` has the same dimensionality as the input.
    """
    image_stack, input_was_2d = _as_image_stack(data_array, name="data_array")
    n_images = image_stack.shape[0]

    if not (0 <= image_index < n_images):
        raise ValueError(
            f"image_index={image_index} is out of bounds for {n_images} image(s)."
        )

    mean_image = np.nanmean(image_stack, axis=0)
    std_image = np.nanstd(image_stack, axis=0)

    print(f"Removing hot pixels from {n_images} image(s)...")

    clean_list = []
    n_removed_list = []
    pct_removed_list = []

    for image in image_stack:
        result = _remove_xrays(
            image=image,
            mean_image=mean_image,
            std_image=std_image,
            std_factor=std_factor,
        )
        clean_list.append(result["clean_image"])
        n_removed_list.append(result["n_removed"])
        pct_removed_list.append(result["pct_removed"])

    clean_stack = np.stack(clean_list)
    n_removed = np.asarray(n_removed_list, dtype=int)
    pct_removed = np.asarray(pct_removed_list, dtype=float)

    clean_data = _restore_image_dimensionality(clean_stack, input_was_2d)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)

        axes[0].plot(np.arange(n_images), pct_removed)
        axes[0].set_title("Percent Pixels Removed")
        axes[0].set_xlabel("Image Number")
        axes[0].set_ylabel("Percent")

        im1 = axes[1].imshow(image_stack[image_index], cmap="jet")
        axes[1].set_title("Original Image")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(clean_stack[image_index], cmap="jet")
        axes[2].set_title("Cleaned Image")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    if return_dict:
        return {
            "clean_data": clean_data,
            "pct_removed": pct_removed,
            "n_removed": n_removed,
            "mean_image": mean_image,
            "std_image": std_image,
            "input_was_2d": input_was_2d,
        }

    return clean_data


def remove_xrays_pool(
    data_array,
    std_factor=STD_FACTOR,
    plot=False,
    image_index=0,
    return_dict=True,
    max_workers=MAX_PROCESSORS,
    progress_interval=100,
):
    """
    Remove hot pixels from one image or a stack of images in parallel using a
    threshold based on the stack mean and standard deviation.

    This function converts the input to a 3D image stack internally, computes
    the stack mean and standard deviation, removes hot pixels from each image
    in parallel, and then restores the original dimensionality before returning.

    Parameters
    ----------
    data_array : np.ndarray
        Input image data, either:
        - 2D: (rows, cols)
        - 3D: (n_images, rows, cols)
    std_factor : float, optional
        Threshold multiplier used to identify hot pixels.
    plot : bool, optional
        If True, plot:
        - percent of removed pixels vs image index
        - one example original image
        - one example cleaned image
    image_index : int, optional
        Image index used for plotting when the input is a stack.
    return_dict : bool, optional
        If True, return a dictionary.
        If False, return only the cleaned data array.
    max_workers : int, optional
        Maximum number of worker threads.
    progress_interval : int, optional
        Print progress every `progress_interval` completed images.

    Returns
    -------
    result : dict or np.ndarray
        If return_dict=True:
            {
                "clean_data": np.ndarray,
                "pct_removed": np.ndarray of shape (n_images,),
                "n_removed": np.ndarray of shape (n_images,),
                "mean_image": np.ndarray of shape (rows, cols),
                "std_image": np.ndarray of shape (rows, cols),
                "input_was_2d": bool,
            }

        If return_dict=False:
            clean_data

        `clean_data` has the same dimensionality as the input.
    """
    image_stack, input_was_2d = _as_image_stack(data_array, name="data_array")
    n_images = image_stack.shape[0]

    if not (0 <= image_index < n_images):
        raise ValueError(
            f"image_index={image_index} is out of bounds for {n_images} image(s)."
        )

    if progress_interval is None or progress_interval <= 0:
        progress_interval = max(1, n_images // 20)

    mean_image = np.nanmean(image_stack, axis=0)
    std_image = np.nanstd(image_stack, axis=0)

    print(f"Removing hot pixels from {n_images} image(s)...")

    results = [None] * n_images

    worker = partial(
        _remove_xrays,
        mean_image=mean_image,
        std_image=std_image,
        std_factor=std_factor,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}

        for idx in range(n_images):
            future = executor.submit(worker, image_stack[idx])
            future_to_idx[future] = idx

        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

            completed += 1
            if completed % progress_interval == 0 or completed == n_images:
                print(
                    f"  Completed {completed}/{n_images} "
                    f"({100 * completed / n_images:.1f}%)"
                )

    print("Done removing hot pixels.")

    clean_stack = np.stack([result["clean_image"] for result in results])
    n_removed = np.asarray([result["n_removed"] for result in results], dtype=int)
    pct_removed = np.asarray([result["pct_removed"] for result in results], dtype=float)

    clean_data = _restore_image_dimensionality(clean_stack, input_was_2d)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)

        axes[0].plot(np.arange(n_images), pct_removed)
        axes[0].set_title("Percent Pixels Removed")
        axes[0].set_xlabel("Image Number")
        axes[0].set_ylabel("Percent")

        im1 = axes[1].imshow(image_stack[image_index], cmap="jet")
        axes[1].set_title("Original Image")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(clean_stack[image_index], cmap="jet")
        axes[2].set_title("Cleaned Image")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    if return_dict:
        return {
            "clean_data": clean_data,
            "pct_removed": pct_removed,
            "n_removed": n_removed,
            "mean_image": mean_image,
            "std_image": std_image,
            "input_was_2d": input_was_2d,
        }

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
    figsize=(15, 4),
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
    center_guess=(CENTER_Y, CENTER_X),
    search_radius=20,
    center_mask=None,
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
    **kwargs,
):
    """Find diffraction centers for one image or a stack of images in parallel."""
    if "mask" in kwargs and center_mask is None:
        center_mask = kwargs.pop("mask")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

    image_stack, input_was_2d = _as_image_stack(data_array, name="data_array")
    n_images = image_stack.shape[0]

    print(f"Finding centers for {n_images} images...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx in range(n_images):
            future = executor.submit(
                _center_worker,
                idx=idx,
                image=image_stack[idx],
                center_guess=center_guess,
                search_radius=search_radius,
                mask=center_mask,
                r_min=r_min,
                r_max=r_max,
                downsample=downsample,
                intensity_threshold=intensity_threshold,
                top_percentile=top_percentile,
            )
            future_to_idx[future] = idx

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

    center_y = np.array([d["center_y"] for d in results_list], dtype=float)
    center_x = np.array([d["center_x"] for d in results_list], dtype=float)
    centers_xy = np.column_stack((center_x, center_y))
    score = np.array([d["score"] for d in results_list], dtype=float)
    image_index = np.arange(n_images)
    per_image_results = [d["full_result"] for d in results_list]

    if plot_example:
        if not (0 <= example_index < n_images):
            raise ValueError("example_index is out of bounds.")

        example_result = per_image_results[example_index]
        img = image_stack[example_index].astype(float)
        img_plot = np.nan_to_num(img, nan=0.0)

        fig, axes = plt.subplots(1, 3, figsize=figsize_example)

        axes[0].imshow(img_plot, cmap="inferno")
        axes[0].plot(center_guess[1], center_guess[0], "co", label="Fixed guess")
        axes[0].plot(center_x[example_index], center_y[example_index], "r+", ms=12, mew=2, label="Best")
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
        axes[1].plot(center_x[example_index], center_y[example_index], "r+", ms=12, mew=2, label="Best")
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
        "centers_xy": centers_xy,
        "score": score,
        "image_index": image_index,
        "per_image_results": per_image_results,
        "input_was_2d": input_was_2d,
    }


def normalize_centers(centers, n_images, use_average_center=False):
    """Backward-compatible wrapper for center normalization in (x, y) order."""
    return _normalize_centers_xy(centers, n_images, use_average_center=use_average_center)


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
    nan_radial_range,
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
    image : np.ndarray
        2D diffraction image.
    center : array-like
        (x_center, y_center) for this image.
    npt : int
        Number of radial bins.
    unit : str
        Radial unit for integration.
    radial_range : tuple or None
        Radial range passed directly to pyFAI.integrate1d.
        This truncates the returned profile.
    nan_radial_range : tuple or None
        Range to keep in the returned profile. Values outside this range
        are replaced with NaN after integration. This preserves array length.
        Example:
        - (0.3, None): set radial < 0.3 to NaN
        - (0.3, 4.5): keep only 0.3 <= radial <= 4.5
    azimuth_range : tuple or None
        Azimuthal range passed to pyFAI.
    mask : np.ndarray or None
        Boolean image mask where True means excluded pixel.
    dark : np.ndarray or None
        Dark image for correction.
    flat : np.ndarray or None
        Flat-field image for correction.
    polarization_factor : float or None
        Polarization correction factor.
    method : str or tuple
        Integration method passed to pyFAI.
    pixel1, pixel2 : float
        Pixel sizes in meters.
    distance : float
        Sample-detector distance in meters.
    wavelength : float
        Beam wavelength in meters.
    tilt_angle : float
        Detector tilt angle in radians.
    tilt_plane_rotation : float
        Tilt-plane rotation in radians.
    rot3 : float
        In-plane detector rotation in radians.
    error_mode : {"raise", "warn", "ignore"}
        Error handling mode.

    Returns
    -------
    dict
        Dictionary containing:
        - "index" : int
        - "radial" : np.ndarray or None
        - "intensity" : np.ndarray or None
        - "success" : bool
        - "error" : str or None
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

        intensity = np.asarray(intensity, dtype=float)

        # Post-integration masking in radial space
        if nan_radial_range is not None:
            if len(nan_radial_range) != 2:
                raise ValueError("nan_radial_range must be a tuple: (rmin, rmax)")

            rmin_nan, rmax_nan = nan_radial_range

            keep_mask = np.ones_like(radial, dtype=bool)
            if rmin_nan is not None:
                keep_mask &= radial >= rmin_nan
            if rmax_nan is not None:
                keep_mask &= radial <= rmax_nan

            intensity = intensity.copy()
            intensity[~keep_mask] = np.nan

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
    nan_radial_range=None,
    azimuth_range=None,
    integration_mask=None,
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
    **kwargs,
):
    """
    Compute azimuthal averages for one image or a stack of images using pyFAI.

    Parameters
    ----------
    images : np.ndarray
        2D image or 3D image stack.
    centers : tuple or np.ndarray
        Center(s) in (x, y) pixel coordinates.
    use_average_center : bool, optional
        If True and centers are provided per-image, average them and use one
        center for all images.
    npt : int, optional
        Number of radial bins.
    unit : str, optional
        Radial unit for pyFAI.
    radial_range : tuple or None, optional
        Range passed directly to pyFAI.integrate1d. This truncates the output.
    nan_radial_range : tuple or None, optional
        Radial range to keep after integration. Values outside this range are
        set to NaN while preserving profile length.
        Example:
        - (0.3, None): mask low-Q region below 0.3
        - (0.3, 4.5): keep only 0.3 <= Q <= 4.5
    azimuth_range : tuple or None, optional
        Azimuthal integration range.
    integration_mask : np.ndarray or None, optional
        Boolean image mask where True means excluded pixel.
    dark, flat : np.ndarray or None, optional
        Dark and flat-field correction images.
    polarization_factor : float or None, optional
        Polarization correction factor.
    method : str or tuple, optional
        Integration method passed to pyFAI.
    pixel1, pixel2 : float, optional
        Pixel sizes in meters.
    distance : float, optional
        Sample-detector distance in meters.
    wavelength : float, optional
        Beam wavelength in meters.
    tilt_angle, tilt_plane_rotation, rot3 : float, optional
        Detector geometry parameters.
    return_dict : bool, optional
        If True, return dictionary.
    error_mode : {"raise", "warn", "ignore"}, optional
        Error handling mode.
    max_workers : int or None, optional
        Number of worker threads.
    progress_interval : int, optional
        Print progress every `progress_interval` completed images.

    Returns
    -------
    dict or tuple
        If return_dict=True:
            {
                "radial": np.ndarray,
                "profiles": np.ndarray,
                "centers_used": np.ndarray,
                "success": np.ndarray,
                "geometry": dict,
                "unit": str,
                "input_was_2d": bool,
                "radial_range": tuple or None,
                "nan_radial_range": tuple or None,
            }
        Otherwise returns `(radial, profile)` for 2D input or
        `(radial, profiles)` for 3D input.
    """
    if "mask" in kwargs and integration_mask is None:
        integration_mask = kwargs.pop("mask")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

    image_stack, input_was_2d = _as_image_stack(images, name="images")
    n_images = image_stack.shape[0]

    centers_used = _normalize_centers_xy(
        centers,
        n_images=n_images,
        use_average_center=use_average_center,
    )

    if integration_mask is not None:
        integration_mask = np.asarray(integration_mask, dtype=bool)
        if integration_mask.shape != image_stack.shape[1:]:
            raise ValueError(
                f"integration_mask shape {integration_mask.shape} does not match image shape {image_stack.shape[1:]}"
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
                image=image_stack[idx],
                center=centers_used[idx],
                npt=npt,
                unit=unit,
                radial_range=radial_range,
                nan_radial_range=nan_radial_range,
                azimuth_range=azimuth_range,
                mask=integration_mask,
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
            "input_was_2d": input_was_2d,
            "radial_range": radial_range,
            "nan_radial_range": nan_radial_range,
        }

    if input_was_2d:
        return radial_out, profiles[0]

    return radial_out, profiles


def compute_background_azimuthal_average(
    background_input,
    centers=None,
    center_guess=(CENTER_Y, CENTER_X),
    compute_center_if_missing=True,
    center_from="mean",
    search_radius=20,
    center_mask=None,
    r_min=0,
    r_max=None,
    downsample=1,
    intensity_threshold=None,
    top_percentile=60,
    npt=5000,
    radial_range=None,
    nan_radial_range=None,
    azimuth_range=None,
    integration_mask=None,
    unit="q_A^-1",
    method=("bbox", "csr", "cython"),
    polarization_factor=POLARIZATION_FACTOR,
    dark=DARK,
    flat=FLAT,
    plot=False,
    image_index=0,
    figsize=FIGSIZE,
    return_dict=True,
):
    """
    Compute azimuthal averages for background diffraction image(s), with optional
    automatic determination of the diffraction center.

    This function supports both single 2D images and 3D stacks of background images.
    It can either use user-provided centers or determine centers automatically
    from the background data using a radial-profile sharpness method.

    Parameters
    ----------
    background_input : np.ndarray or dict
        Background image data. Can be:
        - 2D array of shape (rows, cols)
        - 3D array of shape (n_images, rows, cols)
        - dict containing:
            - "background_stack": 3D array
            - or "background_mean": 2D array
    centers : array-like or None, optional
        Beam center(s) in (x, y) pixel coordinates.
        Accepts:
        - (2,) → single center applied to all images
        - (n_images, 2) → per-image centers
        If None, centers will be computed if `compute_center_if_missing=True`.
    center_guess : tuple, optional
        Initial guess for center finding as (y, x).
    compute_center_if_missing : bool, optional
        If True and `centers` is None, automatically determine centers.
    center_from : {"mean", "each"}, optional
        Strategy for automatic center determination:
        - "mean": compute center from mean background image and apply to all images
        - "each": compute center independently for each image
    search_radius : int, optional
        Pixel radius around the center_guess to search for the optimal center.
    center_mask : np.ndarray or None, optional
        Boolean mask used during center finding (True = excluded pixel).
    r_min : int, optional
        Minimum radius (in pixels) used when evaluating radial profiles.
    r_max : int or None, optional
        Maximum radius (in pixels) used when evaluating radial profiles.
    downsample : int, optional
        Downsampling factor for center finding. Must be >= 1.
    intensity_threshold : float or None, optional
        Minimum intensity threshold for selecting pixels during center finding.
    top_percentile : float or None, optional
        Use only pixels above this percentile for center finding.
    npt : int, optional
        Number of radial bins for azimuthal integration.
    radial_range : tuple or None, optional
        Radial range passed directly to pyFAI.integrate1d.
        This truncates the returned radial axis and profile.
    nan_radial_range : tuple or None, optional
        Radial range to keep after integration. Values outside this range are
        replaced with NaN while preserving the returned profile length.
        This is useful for masking out unreliable low-Q or high-Q regions
        such as the beamstop region.

        Examples:
        - (0.3, None): set radial values below 0.3 to NaN
        - (0.3, 4.5): keep only 0.3 <= radial <= 4.5, set the rest to NaN
    azimuth_range : tuple or None, optional
        (min_angle, max_angle) azimuthal range in degrees.
    integration_mask : np.ndarray or None, optional
        Boolean mask applied during azimuthal integration (True = excluded).
    unit : str, optional
        Radial unit for output (e.g., "q_A^-1", "2th_deg", "r_mm").
    method : tuple or str, optional
        Integration method passed to pyFAI.
    polarization_factor : float, optional
        Polarization correction factor.
    dark : np.ndarray or None, optional
        Dark current image for correction.
    flat : np.ndarray or None, optional
        Flat-field correction image.
    plot : bool, optional
        If True, display:
        - an example background image
        - its azimuthal profile
        - the mean background profile
    image_index : int, optional
        Index of image to use for plotting if multiple images are present.
    figsize : tuple, optional
        Figure size for plotting.
    return_dict : bool, optional
        If True, return results as a dictionary.
        If False, return tuple.

    Returns
    -------
    result : dict or tuple
        If return_dict=True:
            {
                "radial": np.ndarray of shape (npt,),
                "background_profiles": np.ndarray of shape (n_images, npt),
                "background_profile_mean": np.ndarray of shape (npt,),
                "background_profile_std": np.ndarray of shape (npt,),
                "background_images_used": np.ndarray,
                "centers_used": np.ndarray of shape (n_images, 2),
                "center_result": dict or None,
                "pyfai_result": dict,
                "input_was_2d": bool,
                "radial_range": tuple or None,
                "nan_radial_range": tuple or None,
            }

        If return_dict=False:
            (radial, background_profiles, background_profile_mean)

    Raises
    ------
    ValueError
        If:
        - background_input is invalid
        - centers are missing and cannot be computed
        - center_from is invalid
        - image_index is out of bounds

    Notes
    -----
    - Centers are internally handled in (x, y) format for pyFAI compatibility.
    - If `center_from="mean"`, the center is computed once from the mean image
      and applied to all images.
    - If `center_from="each"`, center finding is parallelized across images.
    - `radial_range` truncates the integration domain.
    - `nan_radial_range` preserves profile length and replaces excluded radial
      values with NaN after integration.
    - The function preserves input dimensionality information via
      `input_was_2d`.
    """
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

    background_images, input_was_2d = _as_image_stack(background_images, name="background_input")
    n_bg = background_images.shape[0]

    if not (0 <= image_index < n_bg):
        raise ValueError(
            f"image_index={image_index} is out of bounds for {n_bg} background image(s)."
        )

    center_result = None

    if centers is not None:
        centers_array = _normalize_centers_xy(centers, n_bg, use_average_center=False)
    else:
        if not compute_center_if_missing:
            raise ValueError("centers is None and compute_center_if_missing=False.")

        if center_from == "mean":
            mean_bg = np.nanmean(background_images, axis=0)

            center_result = find_diffraction_center_from_guess_radial_fast(
                image=mean_bg,
                center_guess=center_guess,
                search_radius=search_radius,
                mask=center_mask,
                r_min=r_min,
                r_max=r_max,
                downsample=downsample,
                intensity_threshold=intensity_threshold,
                top_percentile=top_percentile,
                plot=False,
            )

            centers_array = np.tile(
                np.array([center_result["center_x"], center_result["center_y"]], dtype=float),
                (n_bg, 1),
            )

        elif center_from == "each":
            center_result = find_centers_in_stack_radial_parallel(
                data_array=background_images,
                center_guess=center_guess,
                center_mask=center_mask,
                search_radius=search_radius,
                r_min=r_min,
                r_max=r_max,
                downsample=downsample,
                intensity_threshold=intensity_threshold,
                top_percentile=top_percentile,
                progress_interval=10,
                max_workers=MAX_PROCESSORS,
            )
            centers_array = center_result["centers_xy"]

        else:
            raise ValueError("center_from must be 'mean' or 'each'")

    pyfai_result = azimuthal_average_pyfai(
        images=background_images,
        centers=centers_array,
        npt=npt,
        radial_range=radial_range,
        nan_radial_range=nan_radial_range,
        azimuth_range=azimuth_range,
        integration_mask=integration_mask,
        unit=unit,
        method=method,
        polarization_factor=polarization_factor,
        dark=dark,
        flat=flat,
        return_dict=True,
    )

    radial = pyfai_result["radial"]
    background_profiles = pyfai_result["profiles"]
    background_profile_mean = np.nanmean(background_profiles, axis=0)
    background_profile_std = np.nanstd(background_profiles, axis=0)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        im = axes[0].imshow(background_images[image_index], cmap="jet")
        axes[0].set_title("Background Image")
        axes[0].set_xlabel("Pixel")
        axes[0].set_ylabel("Pixel")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        axes[1].plot(
            radial,
            background_profiles[image_index],
            label="Example Background Profile",
        )
        axes[1].plot(
            radial,
            background_profile_mean,
            label="Mean Background Profile",
            linewidth=2,
        )
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
            "input_was_2d": input_was_2d,
            "radial_range": radial_range,
            "nan_radial_range": nan_radial_range,
        }

    return radial, background_profiles, background_profile_mean


def subtract_scaled_background_profile(
    radial,
    profiles,
    background_profile,
    norm_range,
    mode="mean",
    scale_method="ratio",
    plot=False,
    plot_scale_factors=False,
    plot_indices=None,
    figsize=FIGSIZE,
    alpha=0.8,
    return_dict=True,
):
    """
    Subtract a scaled 1D background profile from azimuthally averaged profiles.

    For each input profile, a scalar background scale factor is computed from
    a specified radial range, and the scaled background profile is subtracted:

        corrected_profile_i = profile_i - scale_factor_i * background_profile

    This function is intended for use after azimuthal integration, where both
    the diffraction data and the background have already been reduced to 1D
    profiles on the same radial axis.

    Parameters
    ----------
    radial : np.ndarray
        1D radial axis of shape (n_q,).
    profiles : np.ndarray
        Input 1D profiles, either:
        - 1D array of shape (n_q,)
        - 2D array of shape (n_profiles, n_q)
    background_profile : np.ndarray
        1D background profile of shape (n_q,).
    norm_range : tuple
        Radial range used to determine the background scale factor:
        (r_min, r_max)
    mode : {"mean", "sum", "median", "max"}, optional
        Statistic used when `scale_method="ratio"`.
        Ignored when `scale_method="least_squares"`.
    scale_method : {"ratio", "least_squares"}, optional
        Method used to compute the scale factor for each profile:
        - "ratio":
            scale_factor = statistic(profile in norm_range) /
                           statistic(background_profile in norm_range)
        - "least_squares":
            scale_factor is the best-fit scalar minimizing the residual
            between the profile and background in the normalization region.
    plot : bool, optional
        If True, plot selected original profiles, scaled background profiles,
        and corrected profiles.
    plot_scale_factors : bool, optional
        If True, plot the background scale factor versus profile index.
    plot_indices : None, int, or sequence of int, optional
        Which profiles to plot if `plot=True`.
        If None, plots the first profile.
    figsize : tuple, optional
        Figure size for plotting.
    alpha : float, optional
        Line transparency for profile plots.
    return_dict : bool, optional
        If True, return a dictionary.
        If False, return a tuple.

    Returns
    -------
    result : dict or tuple
        If return_dict=True:
            {
                "corrected_profiles": np.ndarray,
                "scale_factors": np.ndarray,
                "background_profile": np.ndarray,
                "scaled_background_profiles": np.ndarray,
                "normalization_mask": np.ndarray,
                "norm_range": tuple,
                "mode": str,
                "scale_method": str,
                "input_was_1d": bool,
            }

        If return_dict=False:
            (corrected_profiles, scale_factors)

        If the input `profiles` was 1D, the returned `corrected_profiles`
        will also be 1D. Otherwise it will be 2D.

    Raises
    ------
    ValueError
        If the input shapes are incompatible, if `norm_range` is invalid,
        if no radial points fall within the normalization range, or if the
        scale factors cannot be computed.
    """
    radial = np.asarray(radial, dtype=float)
    background_profile = np.asarray(background_profile, dtype=float)
    profiles = np.asarray(profiles, dtype=float)

    if radial.ndim != 1:
        raise ValueError("radial must be 1D.")
    if background_profile.ndim != 1:
        raise ValueError("background_profile must be 1D.")
    if background_profile.shape[0] != radial.shape[0]:
        raise ValueError("background_profile must have the same length as radial.")

    if profiles.ndim == 1:
        profiles_2d = profiles[None, :]
        input_was_1d = True
    elif profiles.ndim == 2:
        profiles_2d = profiles
        input_was_1d = False
    else:
        raise ValueError("profiles must be 1D or 2D.")

    if profiles_2d.shape[1] != radial.shape[0]:
        raise ValueError("profiles.shape[-1] must match len(radial).")

    if norm_range is None or len(norm_range) != 2:
        raise ValueError("norm_range must be a tuple: (r_min, r_max).")

    r_min, r_max = norm_range
    if r_min >= r_max:
        raise ValueError("norm_range must satisfy r_min < r_max.")

    norm_mask = (radial >= r_min) & (radial <= r_max)
    if not np.any(norm_mask):
        raise ValueError("No radial points fall inside norm_range.")

    profile_region = profiles_2d[:, norm_mask]
    background_region = background_profile[norm_mask]

    if np.any(~np.isfinite(background_region)):
        raise ValueError("background_profile contains non-finite values in norm_range.")

    # ------------------------------------------------------------
    # Compute scale factors
    # ------------------------------------------------------------
    if scale_method == "ratio":

        if mode == "mean":
            profile_vals = np.nanmean(profile_region, axis=1)
            bg_val = np.nanmean(background_region)
        elif mode == "sum":
            profile_vals = np.nansum(profile_region, axis=1)
            bg_val = np.nansum(background_region)
        elif mode == "median":
            profile_vals = np.nanmedian(profile_region, axis=1)
            bg_val = np.nanmedian(background_region)
        elif mode == "max":
            profile_vals = np.nanmax(profile_region, axis=1)
            bg_val = np.nanmax(background_region)
        else:
            raise ValueError("mode must be one of: 'mean', 'sum', 'median', 'max'.")

        if not np.isfinite(bg_val) or bg_val == 0:
            raise ValueError("Background normalization value is zero or non-finite.")

        scale_factors = profile_vals / bg_val

    elif scale_method == "least_squares":
        denom = np.nansum(background_region ** 2)
        if not np.isfinite(denom) or denom == 0:
            raise ValueError("Background least-squares denominator is zero or non-finite.")

        scale_factors = np.array([
            np.nansum(profile_region[i] * background_region) / denom
            for i in range(profiles_2d.shape[0])
        ], dtype=float)

    else:
        raise ValueError("scale_method must be 'ratio' or 'least_squares'.")

    if np.any(~np.isfinite(scale_factors)):
        raise ValueError("Some scale factors are not finite.")

    # ------------------------------------------------------------
    # Subtract scaled background
    # ------------------------------------------------------------
    scaled_background_profiles = scale_factors[:, None] * background_profile[None, :]
    corrected_profiles_2d = profiles_2d - scaled_background_profiles

    # ------------------------------------------------------------
    # Plot scale factors
    # ------------------------------------------------------------
    if plot_scale_factors:
        plt.figure(figsize=figsize)
        plt.plot(np.arange(len(scale_factors)), scale_factors, lw=1.5)
        plt.axhline(1.0, color="k", linestyle="--", alpha=0.6)
        plt.xlabel("Profile Index")
        plt.ylabel("Background Scale Factor")
        plt.title("Background Scale Factor vs Profile Index")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # Plot example profiles
    # ------------------------------------------------------------
    if plot:
        n_profiles = profiles_2d.shape[0]

        if plot_indices is None:
            plot_indices = [0]
        elif np.isscalar(plot_indices):
            plot_indices = [int(plot_indices)]
        else:
            plot_indices = [int(i) for i in plot_indices]

        for idx in plot_indices:
            if idx < 0 or idx >= n_profiles:
                raise ValueError(f"plot index {idx} is out of bounds for {n_profiles} profiles.")

        fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)

        for idx in plot_indices:
            axes[0].plot(radial, profiles_2d[idx], alpha=alpha, label=f"Profile {idx}")
        axes[0].axvspan(r_min, r_max, color="gray", alpha=0.2)
        axes[0].set_title("Original Profiles")
        axes[0].set_xlabel("Radial coordinate")
        axes[0].set_ylabel("Intensity")
        axes[0].legend()

        for idx in plot_indices:
            axes[1].plot(
                radial,
                scaled_background_profiles[idx],
                alpha=alpha,
                label=f"Scaled BG {idx}",
            )
        axes[1].axvspan(r_min, r_max, color="gray", alpha=0.2)
        axes[1].set_title("Scaled Background Profiles")
        axes[1].set_xlabel("Radial coordinate")
        axes[1].legend()

        for idx in plot_indices:
            axes[2].plot(
                radial,
                corrected_profiles_2d[idx],
                alpha=alpha,
                label=f"Corrected {idx}",
            )
        axes[2].axvspan(r_min, r_max, color="gray", alpha=0.2)
        axes[2].set_title("Background-Subtracted Profiles")
        axes[2].set_xlabel("Radial coordinate")
        axes[2].legend()

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # Restore original dimensionality
    # ------------------------------------------------------------
    if input_was_1d:
        corrected_profiles = corrected_profiles_2d[0]
        scaled_background = scaled_background_profiles[0]
    else:
        corrected_profiles = corrected_profiles_2d
        scaled_background = scaled_background_profiles

    if return_dict:
        return {
            "corrected_profiles": corrected_profiles,
            "scale_factors": scale_factors,
            "background_profile": background_profile,
            "scaled_background_profiles": scaled_background,
            "normalization_mask": norm_mask,
            "norm_range": norm_range,
            "mode": mode,
            "scale_method": scale_method,
            "input_was_1d": input_was_1d,
        }

    return corrected_profiles, scale_factors


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


def _als_baseline_1d(y, lam, p, niter):
    """
    Internal helper: compute ALS baseline for a single 1D array.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    finite_mask = np.isfinite(y)
    if np.sum(finite_mask) < 3:
        return np.full_like(y, np.nan)

    # Fill NaNs by interpolation for fitting only
    if not np.all(finite_mask):
        x = np.arange(n)
        y_fit = np.interp(x, x[finite_mask], y[finite_mask])
    else:
        y_fit = y.copy()

    # Smoothness operator
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n), format="csc")
    DTD = D.T @ D

    w = np.ones(n)

    for _ in range(niter):
        W = sparse.spdiags(w, 0, n, n)
        Z = W + lam * DTD
        z = spsolve(Z, w * y_fit)
        w = p * (y_fit > z) + (1 - p) * (y_fit <= z)

    return z


def subtract_als_baseline(
    data_array,
    lam=1e6,
    p=0.01,
    niter=10,
    plot=False,
    profile_index=0,
    x_vals=None,
    figsize=FIGSIZE,
    return_dict=True,
):
    """
    Subtract a smooth baseline using asymmetric least squares (ALS).

    This method is well-suited for diffraction or spectroscopy data with
    sharp positive peaks on top of a slowly varying background. It estimates
    a smooth baseline that follows the lower envelope of the data.

    Parameters
    ----------
    data_array : np.ndarray
        Input data:
        - 1D array (n_x,)
        - 2D array (n_profiles, n_x)
    lam : float, optional
        Smoothness parameter. Larger values → smoother baseline.
        Typical: 1e4 – 1e8.
    p : float, optional
        Asymmetry parameter (0 < p < 1).
        Smaller values → baseline stays below peaks (recommended ~0.001–0.05).
    niter : int, optional
        Number of ALS iterations.
    plot : bool, optional
        If True, plot one example profile and its baseline.
    profile_index : int, optional
        Index to plot if input is 2D.
    x_vals : np.ndarray or None, optional
        Optional x-axis values (e.g. q). If None, pixel index is used.
    figsize : tuple, optional
        Figure size for plotting.
    return_dict : bool, optional
        If True, return dictionary. If False, return tuple.

    Returns
    -------
    result : dict or tuple
        If return_dict=True:
            {
                "corrected_data": np.ndarray,
                "baselines": np.ndarray,
                "lam": float,
                "p": float,
                "niter": int,
                "input_was_2d": bool,
            }

        If return_dict=False:
            (corrected_data, baselines)

    Raises
    ------
    ValueError
        If parameters are invalid or dimensions mismatch.

    Notes
    -----
    - NaNs are ignored during fitting (interpolated internally).
    - Output preserves input dimensionality.
    - Recommended starting values for diffraction data:
        lam = 1e6, p = 0.01
    """
    if lam <= 0:
        raise ValueError("lam must be positive.")
    if not (0 < p < 1):
        raise ValueError("p must satisfy 0 < p < 1.")
    if niter < 1:
        raise ValueError("niter must be >= 1.")

    data_array = np.asarray(data_array, dtype=float)

    if data_array.ndim == 1:
        data_stack = data_array[None, :]
        input_was_1d = True
    elif data_array.ndim == 2:
        data_stack = data_array
        input_was_1d = False
    else:
        raise ValueError("data_array must be 1D or 2D (profiles).")

    n_profiles, n_x = data_stack.shape

    baselines = []
    for i in range(n_profiles):
        baseline = _als_baseline_1d(data_stack[i], lam, p, niter)
        baselines.append(baseline)

    baselines = np.asarray(baselines, dtype=float)
    corrected_stack = data_stack - baselines

    # Restore original dimensionality
    if input_was_1d:
        corrected_out = corrected_stack[0]
        baselines_out = baselines[0]
    else:
        corrected_out = corrected_stack
        baselines_out = baselines

    # Plot
    if plot:
        if input_was_1d:
            y_plot = data_stack[0]
            baseline_plot = baselines[0]
            corrected_plot = corrected_stack[0]
            title_suffix = ""
        else:
            if not (0 <= profile_index < n_profiles):
                raise ValueError(
                    f"profile_index={profile_index} out of bounds for {n_profiles} profiles."
                )
            y_plot = data_stack[profile_index]
            baseline_plot = baselines[profile_index]
            corrected_plot = corrected_stack[profile_index]
            title_suffix = f" (Profile {profile_index})"

        if x_vals is None:
            x_plot = np.arange(n_x)
        else:
            x_plot = np.asarray(x_vals)
            if len(x_plot) != n_x:
                raise ValueError("x_vals must match data length.")

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)

        axes[0].plot(x_plot, y_plot, label="Original")
        axes[0].plot(x_plot, baseline_plot, label="ALS baseline")
        axes[0].set_title(f"Original + Baseline{title_suffix}")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("Intensity")
        axes[0].legend()

        axes[1].plot(x_plot, corrected_plot)
        axes[1].set_title(f"Baseline Subtracted{title_suffix}")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("Corrected intensity")

        plt.tight_layout()
        plt.show()

    if return_dict:
        return {
            "corrected_data": corrected_out,
            "baselines": baselines_out,
            "lam": lam,
            "p": p,
            "niter": niter,
            "input_was_1d": input_was_1d,
        }

    return corrected_out, baselines_out


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
    figsize=(8, 5),
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


def compute_difference_pdf(
    q,
    delta_iq,
    r_max=20.0,
    n_r=2000,
    q_range=None,
    window="lorch",
    plot=False,
    profile_index=0,
    figsize=FIGSIZE,
    return_dict=True,
):
    """
    Compute a difference PDF-like signal dG(r) from 1D difference scattering
    data using a sine Fourier transform.

    This function transforms either a single 1D difference profile or a 2D
    stack of difference profiles from reciprocal space into real space using

        dG(r) = (2 / pi) * integral[ Q * dI(Q) * M(Q) * sin(Qr) dQ ]

    where M(Q) is an optional modification/window function such as the Lorch
    function.

    The implementation is partially vectorized:
    - it loops over profiles in Python
    - but computes all r values at once for each profile using NumPy array math

    This gives a substantial speedup compared with looping over both profiles
    and r values in Python.

    Parameters
    ----------
    q : np.ndarray
        1D Q axis of shape (n_q,), typically in inverse angstroms.
    delta_iq : np.ndarray
        Difference scattering data, either:
        - 1D array of shape (n_q,)
        - 2D array of shape (n_profiles, n_q)

        This may be dI(Q), dI/I(Q), or another difference signal in Q-space.
    r_max : float, optional
        Maximum r value in angstroms for the output transform.
    n_r : int, optional
        Number of points in the output r axis.
    q_range : tuple or None, optional
        (q_min, q_max) range to keep before transforming.
        If None, all finite Q values are used.
    window : {"none", "lorch"}, optional
        Modification function applied before the transform:
        - "none"  : no windowing
        - "lorch" : Lorch modification function
    plot : bool, optional
        If True, plot one example input Q-space profile and the corresponding
        real-space transform.
    profile_index : int, optional
        Which profile to plot if `delta_iq` is 2D.
        Ignored for 1D input.
    figsize : tuple, optional
        Figure size for plotting.
    return_dict : bool, optional
        If True, return a dictionary.
        If False, return (r, dgr).

    Returns
    -------
    result : dict or tuple
        If return_dict=True:
            {
                "r": np.ndarray of shape (n_r,),
                "dgr": np.ndarray,
                "q_used": np.ndarray,
                "delta_iq_used": np.ndarray,
                "window_values": np.ndarray,
                "q_range": tuple or None,
                "window": str,
                "input_was_1d": bool,
            }

        If return_dict=False:
            (r, dgr)

        Output dimensionality matches input dimensionality:
        - 1D input -> 1D dgr
        - 2D input -> 2D dgr

    Raises
    ------
    ValueError
        If input dimensions are invalid, if q and delta_iq do not match in
        length, if q_range is invalid, or if no valid Q points remain after
        masking.

    Notes
    -----
    - This function computes a difference PDF-like signal directly from the
      supplied Q-space data.
    - No atomic or compositional information is required for the transform
      itself.
    - Absolute physical interpretation of the resulting dG(r) depends on the
      normalization of the input delta_iq.
    - The Lorch window reduces termination ripples caused by finite Q range,
      at the cost of some real-space broadening.
    """
    q = np.asarray(q, dtype=float)
    delta_iq = np.asarray(delta_iq, dtype=float)

    if q.ndim != 1:
        raise ValueError("q must be 1D.")

    if delta_iq.ndim == 1:
        delta_2d = delta_iq[None, :]
        input_was_1d = True
    elif delta_iq.ndim == 2:
        delta_2d = delta_iq
        input_was_1d = False
    else:
        raise ValueError("delta_iq must be 1D or 2D.")

    if delta_2d.shape[1] != q.shape[0]:
        raise ValueError("delta_iq.shape[-1] must match len(q).")

    if r_max <= 0:
        raise ValueError("r_max must be positive.")
    if n_r < 2:
        raise ValueError("n_r must be at least 2.")

    # ------------------------------------------------------------
    # Select valid Q range
    # ------------------------------------------------------------
    valid_mask = np.isfinite(q)

    if q_range is not None:
        if len(q_range) != 2:
            raise ValueError("q_range must be a tuple: (q_min, q_max)")
        q_min, q_max = q_range
        if q_min >= q_max:
            raise ValueError("q_range must satisfy q_min < q_max")
        valid_mask &= (q >= q_min) & (q <= q_max)

    if not np.any(valid_mask):
        raise ValueError("No valid Q points remain after applying q_range.")

    q_used = q[valid_mask]
    delta_used = delta_2d[:, valid_mask]

    # ------------------------------------------------------------
    # Build modification/window function
    # ------------------------------------------------------------
    if window == "none":
        window_values = np.ones_like(q_used)

    elif window == "lorch":
        q_max_used = np.nanmax(q_used)
        if not np.isfinite(q_max_used) or q_max_used <= 0:
            raise ValueError("Maximum Q must be positive and finite for Lorch window.")

        x = np.pi * q_used / q_max_used
        window_values = np.ones_like(q_used)
        nonzero = x != 0
        window_values[nonzero] = np.sin(x[nonzero]) / x[nonzero]

    else:
        raise ValueError("window must be one of: 'none', 'lorch'")

    # ------------------------------------------------------------
    # Build r axis
    # ------------------------------------------------------------
    r = np.linspace(0.0, r_max, n_r)
    dgr_2d = np.full((delta_used.shape[0], n_r), np.nan, dtype=float)

    # ------------------------------------------------------------
    # Partially vectorized sine transform
    # ------------------------------------------------------------
    for i in range(delta_used.shape[0]):
        y = np.asarray(delta_used[i], dtype=float)

        finite_mask = np.isfinite(y) & np.isfinite(q_used)
        if np.sum(finite_mask) < 2:
            continue

        q_fit = q_used[finite_mask]
        y_fit = y[finite_mask]
        w_fit = window_values[finite_mask]

        fq = q_fit * y_fit * w_fit

        # Vectorized over all r values at once
        sin_qr = np.sin(np.outer(q_fit, r))  # shape: (n_q_fit, n_r)

        dgr_2d[i] = (2.0 / np.pi) * np.trapezoid(
            fq[:, None] * sin_qr,
            q_fit,
            axis=0,
        )

    # ------------------------------------------------------------
    # Restore original dimensionality
    # ------------------------------------------------------------
    if input_was_1d:
        dgr = dgr_2d[0]
        delta_iq_used = delta_used[0]
    else:
        dgr = dgr_2d
        delta_iq_used = delta_used

        if not (0 <= profile_index < delta_2d.shape[0]):
            raise ValueError(
                f"profile_index={profile_index} is out of bounds for {delta_2d.shape[0]} profile(s)."
            )

    # ------------------------------------------------------------
    # Plot diagnostic example
    # ------------------------------------------------------------
    if plot:
        if input_was_1d:
            q_plot = delta_used[0]
            dgr_plot = dgr_2d[0]
            title_suffix = ""
        else:
            q_plot = delta_used[profile_index]
            dgr_plot = dgr_2d[profile_index]
            title_suffix = f" (Profile {profile_index})"

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].plot(q_used, q_plot, label="Input difference profile")
        axes[0].set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)")
        axes[0].set_ylabel("Difference signal")
        axes[0].set_title(f"Q-space Input{title_suffix}")
        axes[0].legend()

        axes[1].plot(r, dgr_plot, label="dG(r)")
        axes[1].set_xlabel(r"r ($\mathrm{\AA}$)")
        axes[1].set_ylabel("Difference PDF-like signal")
        axes[1].set_title(f"Real-space Transform{title_suffix}")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    if return_dict:
        return {
            "r": r,
            "dgr": dgr,
            "q_used": q_used,
            "delta_iq_used": delta_iq_used,
            "window_values": window_values,
            "q_range": q_range,
            "window": window,
            "input_was_1d": input_was_1d,
        }

    return r, dgr


# X-ray Simulation Functions

## Load x_ray form factors 
def load_form_factor_table(file_path=FORM_FACTOR_FILE):
    """
    Load x-ray form factor coefficients from file.

    Parameters
    ----------
    file_path : str or Path, optional
        Path to form factor coefficient file.

    Returns
    -------
    dict
        Dictionary mapping element symbol -> coefficient list.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Form factor file not found: {file_path}")

    form_factors = {}

    with open(file_path, "r") as f:
        for line in f:
            vals = line.strip().split(",")
            element = vals[0]
            coeffs = [float(val) for val in vals[1:]]
            form_factors[element] = coeffs

    return form_factors

def load_form_factor(element):
    """ 
    Loads x-ray form factor coefficients and returns f(Q) for an element.

    Parameters
    ----------
    element : str
        Element symbol (e.g., "Ba", "Ti", "O")

    Returns
    -------
    ff : callable
        Function that takes Q values and returns f(Q)
    """
    FORM_FACTORS = load_form_factor_table(FORM_FACTOR_FILE)
    try:
        coeffs = FORM_FACTORS[element]
    except KeyError:
        raise ValueError(f"Element '{element}' not found in form factor table.")

    t1 = lambda q: coeffs[0] * np.exp(-coeffs[1] * (q / (4 * np.pi))**2)
    t2 = lambda q: coeffs[2] * np.exp(-coeffs[3] * (q / (4 * np.pi))**2)
    t3 = lambda q: coeffs[4] * np.exp(-coeffs[5] * (q / (4 * np.pi))**2)
    t4 = lambda q: coeffs[6] * np.exp(-coeffs[7] * (q / (4 * np.pi))**2) + coeffs[8]

    return lambda q: t1(q) + t2(q) + t3(q) + t4(q)


def compute_average_form_factors(
    q,
    composition,
    plot=False,
    show_f2=False,
    figsize=FIGSIZE,
    return_dict=True,
):
    """
    Compute composition-weighted average x-ray form factors.

    This function parses a chemical formula (e.g., "BaTiO3") or accepts a
    composition dictionary and computes:

        <f(Q)>   = sum_i c_i f_i(Q)
        <f^2(Q)> = sum_i c_i f_i(Q)^2

    where c_i are atomic fractions.

    Parameters
    ----------
    q : np.ndarray
        1D array of Q values (Å⁻¹).
    composition : str or dict
        Chemical composition, either:
        - string formula (e.g., "BaTiO3")
        - dict of element counts (e.g., {"Ba": 1, "Ti": 1, "O": 3})
    plot : bool, optional
        If True, plot the weighted elemental contributions c_i f_i(Q) along with
        the composition-weighted average <f(Q)>.
    show_f2 : bool, optional
        If True and plot=True, also plot <f^2(Q)> in a second panel.
    figsize : tuple, optional
        Figure size for plotting.
    return_dict : bool, optional
        If True, return results as a dictionary.
        If False, return (f_avg, f2_avg).

    Returns
    -------
    result : dict or tuple
        If return_dict=True:
            {
                "q": np.ndarray,
                "f_avg": np.ndarray,
                "f2_avg": np.ndarray,
                "composition_dict": dict,
                "atomic_fractions": dict,
                "element_form_factors": dict,
                "weighted_element_form_factors": dict,
            }

        If return_dict=False:
            (f_avg, f2_avg)

    Raises
    ------
    ValueError
        If composition cannot be parsed or elements are missing.

    Notes
    -----
    - Neutral atomic form factors are typically used for x-ray total scattering
      normalization, even for ionic compounds.
    - Atomic fractions are computed from the stoichiometric formula.
    """
    q = np.asarray(q, dtype=float)

    if q.ndim != 1:
        raise ValueError("q must be 1D.")

    # ------------------------------------------------------------
    # Parse composition
    # ------------------------------------------------------------
    if isinstance(composition, str):
        tokens = re.findall(r"([A-Z][a-z]?)(\d*)", composition)
        if not tokens:
            raise ValueError(f"Could not parse composition: {composition}")

        composition_dict = {}
        for elem, count in tokens:
            count = int(count) if count else 1
            composition_dict[elem] = composition_dict.get(elem, 0) + count

    elif isinstance(composition, dict):
        composition_dict = composition.copy()

    else:
        raise ValueError("composition must be a string or dict")

    # ------------------------------------------------------------
    # Atomic fractions
    # ------------------------------------------------------------
    total_atoms = sum(composition_dict.values())
    if total_atoms <= 0:
        raise ValueError("Total atom count must be positive.")

    atomic_fractions = {
        elem: count / total_atoms for elem, count in composition_dict.items()
    }

    # ------------------------------------------------------------
    # Compute <f(Q)> and <f^2(Q)>
    # ------------------------------------------------------------
    f_avg = np.zeros_like(q, dtype=float)
    f2_avg = np.zeros_like(q, dtype=float)

    element_form_factors = {}
    weighted_element_form_factors = {}

    for elem, frac in atomic_fractions.items():
        try:
            ff = load_form_factor(elem)
        except KeyError:
            raise ValueError(f"Element '{elem}' not found in form factor table.")
        except ValueError:
            raise ValueError(f"Element '{elem}' not found in form factor table.")

        f_q = ff(q)
        weighted_f_q = frac * f_q

        element_form_factors[elem] = f_q
        weighted_element_form_factors[elem] = weighted_f_q

        f_avg += weighted_f_q
        f2_avg += frac * (f_q ** 2)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    if plot:
        if show_f2:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            ax0, ax1 = axes
        else:
            fig, ax0 = plt.subplots(1, 1, figsize=figsize)

        for elem, weighted_f_q in weighted_element_form_factors.items():
            frac = atomic_fractions[elem]
            ax0.plot(q, weighted_f_q, label=f"{elem} contribution (c={frac:.3f})")

        ax0.plot(q, f_avg, linewidth=2.5, label=r"$\langle f(Q)\rangle$")
        ax0.set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)")
        ax0.set_ylabel("Weighted form factor")
        ax0.set_title("Weighted Elemental Contributions to Average Form Factor")
        ax0.legend()
        ax0.grid(alpha=0.3)

        if show_f2:
            ax1.plot(q, f2_avg, linewidth=2.5, label=r"$\langle f^2(Q)\rangle$")
            ax1.set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)")
            ax1.set_ylabel("Average squared form factor")
            ax1.set_title(r"$\langle f^2(Q)\rangle$")
            ax1.legend()
            ax1.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # Return
    # ------------------------------------------------------------
    if return_dict:
        return {
            "q": q,
            "f_avg": f_avg,
            "f2_avg": f2_avg,
            "composition_dict": composition_dict,
            "atomic_fractions": atomic_fractions,
            "element_form_factors": element_form_factors,
            "weighted_element_form_factors": weighted_element_form_factors,
        }

    return f_avg, f2_avg


def normalize_xray_scattering_to_sq_fq(
    q,
    iq,
    composition,
    mode="total",
    plot=False,
    profile_index=0,
    figsize=FIGSIZE,
    return_dict=True,
):
    """
    Normalize x-ray scattering data to S(Q) and F(Q), or to difference
    quantities ΔS(Q) and ΔF(Q), using composition-weighted atomic form factors.

    This function uses the composition-weighted averages

        <f(Q)>   = sum_i c_i f_i(Q)
        <f^2(Q)> = sum_i c_i f_i(Q)^2

    where c_i are atomic fractions, to compute either:

    Total-scattering normalization:
        S(Q) = 1 + (I(Q) - <f^2(Q)>) / <f(Q)>^2
        F(Q) = Q * (S(Q) - 1)

    Difference-scattering normalization:
        ΔS(Q) = ΔI(Q) / <f(Q)>^2
        ΔF(Q) = Q * ΔS(Q)

    Parameters
    ----------
    q : np.ndarray
        1D array of Q values (Å⁻¹), shape (n_q,).
    iq : np.ndarray
        Input scattering data, either:
        - 1D array of shape (n_q,)
        - 2D array of shape (n_profiles, n_q)

        For mode="total", this should be I(Q), ideally coherent total scattering
        intensity on a compatible scale.

        For mode="difference", this should be ΔI(Q) or another difference
        intensity signal.
    composition : str or dict
        Chemical composition, either:
        - string formula (e.g., "BaTiO3")
        - dict of element counts (e.g., {"Ba": 1, "Ti": 1, "O": 3})
    mode : {"total", "difference"}, optional
        Type of normalization:
        - "total"      : compute S(Q) and F(Q)
        - "difference" : compute ΔS(Q) and ΔF(Q)
    plot : bool, optional
        If True, plot one example normalized profile.
    profile_index : int, optional
        Which profile to plot if iq is 2D.
        Ignored for 1D input.
    figsize : tuple, optional
        Figure size for plotting.
    return_dict : bool, optional
        If True, return a dictionary.
        If False, return a tuple.

    Returns
    -------
    result : dict or tuple
        If mode="total" and return_dict=True:
            {
                "q": np.ndarray,
                "sq": np.ndarray,
                "fq": np.ndarray,
                "f_avg": np.ndarray,
                "f2_avg": np.ndarray,
                "composition_dict": dict,
                "atomic_fractions": dict,
                "input_was_1d": bool,
                "mode": "total",
            }

        If mode="difference" and return_dict=True:
            {
                "q": np.ndarray,
                "delta_sq": np.ndarray,
                "delta_fq": np.ndarray,
                "f_avg": np.ndarray,
                "f2_avg": np.ndarray,
                "composition_dict": dict,
                "atomic_fractions": dict,
                "input_was_1d": bool,
                "mode": "difference",
            }

        If return_dict=False:
            For mode="total":
                (sq, fq)
            For mode="difference":
                (delta_sq, delta_fq)

        Output dimensionality matches input dimensionality:
        - 1D input -> 1D output
        - 2D input -> 2D output

    Raises
    ------
    ValueError
        If q or iq have invalid dimensions, if shapes do not match, if mode is
        invalid, or if <f(Q)> contains invalid or zero values.

    Notes
    -----
    - For mode="total", the formula assumes that iq is on a scale compatible
      with coherent x-ray total scattering intensity.
    - For mode="difference", this gives a practical first-pass normalization
      toward ΔS(Q) and ΔF(Q).
    - Neutral atomic form factors are typically used for x-ray total scattering
      normalization, even for ionic compounds.
    """
    q = np.asarray(q, dtype=float)
    iq = np.asarray(iq, dtype=float)

    if q.ndim != 1:
        raise ValueError("q must be 1D.")

    if iq.ndim == 1:
        iq_2d = iq[None, :]
        input_was_1d = True
    elif iq.ndim == 2:
        iq_2d = iq
        input_was_1d = False
    else:
        raise ValueError("iq must be 1D or 2D.")

    if iq_2d.shape[1] != q.shape[0]:
        raise ValueError("iq.shape[-1] must match len(q).")

    if mode not in ("total", "difference"):
        raise ValueError("mode must be one of: 'total', 'difference'")

    # ------------------------------------------------------------
    # Get average form factors
    # ------------------------------------------------------------
    ff_result = compute_average_form_factors(
        q=q,
        composition=composition,
        plot=False,
        return_dict=True,
    )

    f_avg = np.asarray(ff_result["f_avg"], dtype=float)
    f2_avg = np.asarray(ff_result["f2_avg"], dtype=float)

    denom = f_avg ** 2
    if np.any(~np.isfinite(denom)):
        raise ValueError("<f(Q)>^2 contains non-finite values.")
    if np.any(denom == 0):
        raise ValueError("<f(Q)>^2 contains zeros.")

    # ------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "total":
            sq_2d = 1.0 + (iq_2d - f2_avg[None, :]) / denom[None, :]
            fq_2d = q[None, :] * (sq_2d - 1.0)

        else:  # mode == "difference"
            delta_sq_2d = iq_2d / denom[None, :]
            delta_fq_2d = q[None, :] * delta_sq_2d

    # ------------------------------------------------------------
    # Restore original dimensionality
    # ------------------------------------------------------------
    if input_was_1d:
        if mode == "total":
            sq = sq_2d[0]
            fq = fq_2d[0]
        else:
            delta_sq = delta_sq_2d[0]
            delta_fq = delta_fq_2d[0]
    else:
        if not (0 <= profile_index < iq_2d.shape[0]):
            raise ValueError(
                f"profile_index={profile_index} is out of bounds for {iq_2d.shape[0]} profile(s)."
            )

        if mode == "total":
            sq = sq_2d
            fq = fq_2d
        else:
            delta_sq = delta_sq_2d
            delta_fq = delta_fq_2d

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    if plot:
        if input_was_1d:
            title_suffix = ""
            if mode == "total":
                y1_plot = sq_2d[0]
                y2_plot = fq_2d[0]
            else:
                y1_plot = delta_sq_2d[0]
                y2_plot = delta_fq_2d[0]
        else:
            title_suffix = f" (Profile {profile_index})"
            if mode == "total":
                y1_plot = sq_2d[profile_index]
                y2_plot = fq_2d[profile_index]
            else:
                y1_plot = delta_sq_2d[profile_index]
                y2_plot = delta_fq_2d[profile_index]

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)

        if mode == "total":
            axes[0].plot(q, y1_plot)
            axes[0].set_ylabel("S(Q)")
            axes[0].set_title(f"Normalized S(Q){title_suffix}")

            axes[1].plot(q, y2_plot)
            axes[1].set_ylabel("F(Q)")
            axes[1].set_title(f"Reduced Structure Function F(Q){title_suffix}")

        else:
            axes[0].plot(q, y1_plot)
            axes[0].set_ylabel("ΔS(Q)")
            axes[0].set_title(f"Difference Structure Function ΔS(Q){title_suffix}")

            axes[1].plot(q, y2_plot)
            axes[1].set_ylabel("ΔF(Q)")
            axes[1].set_title(f"Difference Reduced Structure Function ΔF(Q){title_suffix}")

        axes[0].set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)")
        axes[1].set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # Return
    # ------------------------------------------------------------
    if return_dict:
        if mode == "total":
            return {
                "q": q,
                "sq": sq,
                "fq": fq,
                "f_avg": f_avg,
                "f2_avg": f2_avg,
                "composition_dict": ff_result["composition_dict"],
                "atomic_fractions": ff_result["atomic_fractions"],
                "input_was_1d": input_was_1d,
                "mode": mode,
            }

        return {
            "q": q,
            "delta_sq": delta_sq,
            "delta_fq": delta_fq,
            "f_avg": f_avg,
            "f2_avg": f2_avg,
            "composition_dict": ff_result["composition_dict"],
            "atomic_fractions": ff_result["atomic_fractions"],
            "input_was_1d": input_was_1d,
            "mode": mode,
        }

    if mode == "total":
        return sq, fq

    return delta_sq, delta_fq


def compute_delta_gr_from_delta_fq(
    q,
    delta_fq,
    r_max=20.0,
    n_r=2000,
    q_range=None,
    window="lorch",
    plot=False,
    profile_index=0,
    figsize=FIGSIZE,
    return_dict=True,
):
    """
    Compute ΔG(r) from ΔF(Q) using a sine Fourier transform.

    This function assumes the input scattering signal has already been
    normalized to the difference reduced structure function:

        ΔF(Q) = Q ΔS(Q)

    It then computes

        ΔG(r) = (2 / pi) * integral[ ΔF(Q) * M(Q) * sin(Qr) dQ ]

    where M(Q) is an optional modification/window function such as the
    Lorch function.

    The implementation is partially vectorized:
    - it loops over profiles in Python
    - but computes all r values at once for each profile using NumPy array math

    Parameters
    ----------
    q : np.ndarray
        1D Q axis of shape (n_q,), typically in inverse angstroms.
    delta_fq : np.ndarray
        Difference reduced structure function, either:
        - 1D array of shape (n_q,)
        - 2D array of shape (n_profiles, n_q)
    r_max : float, optional
        Maximum r value in angstroms for the output transform.
    n_r : int, optional
        Number of points in the output r axis.
    q_range : tuple or None, optional
        (q_min, q_max) range to keep before transforming.
        If None, all finite Q values are used.
    window : {"none", "lorch"}, optional
        Modification function applied before the transform:
        - "none"  : no windowing
        - "lorch" : Lorch modification function
    plot : bool, optional
        If True, plot one example input ΔF(Q) profile and the corresponding
        real-space transform.
    profile_index : int, optional
        Which profile to plot if `delta_fq` is 2D.
        Ignored for 1D input.
    figsize : tuple, optional
        Figure size for plotting.
    return_dict : bool, optional
        If True, return a dictionary.
        If False, return (r, delta_gr).

    Returns
    -------
    result : dict or tuple
        If return_dict=True:
            {
                "r": np.ndarray of shape (n_r,),
                "delta_gr": np.ndarray,
                "q_used": np.ndarray,
                "delta_fq_used": np.ndarray,
                "window_values": np.ndarray,
                "q_range": tuple or None,
                "window": str,
                "input_was_1d": bool,
            }

        If return_dict=False:
            (r, delta_gr)

        Output dimensionality matches input dimensionality:
        - 1D input -> 1D delta_gr
        - 2D input -> 2D delta_gr

    Raises
    ------
    ValueError
        If input dimensions are invalid, if q and delta_fq do not match in
        length, if q_range is invalid, or if no valid Q points remain after
        masking.

    Notes
    -----
    - This function assumes the input is already ΔF(Q), not raw ΔI(Q).
    - The Lorch window reduces termination ripples caused by finite Q range,
      at the cost of some real-space broadening.
    - This implementation is intended to mirror the standard PDF-style
      sine transform used for difference scattering.
    """
    q = np.asarray(q, dtype=float)
    delta_fq = np.asarray(delta_fq, dtype=float)

    if q.ndim != 1:
        raise ValueError("q must be 1D.")

    if delta_fq.ndim == 1:
        delta_fq_2d = delta_fq[None, :]
        input_was_1d = True
    elif delta_fq.ndim == 2:
        delta_fq_2d = delta_fq
        input_was_1d = False
    else:
        raise ValueError("delta_fq must be 1D or 2D.")

    if delta_fq_2d.shape[1] != q.shape[0]:
        raise ValueError("delta_fq.shape[-1] must match len(q).")

    if r_max <= 0:
        raise ValueError("r_max must be positive.")
    if n_r < 2:
        raise ValueError("n_r must be at least 2.")

    # ------------------------------------------------------------
    # Select valid Q range
    # ------------------------------------------------------------
    valid_mask = np.isfinite(q)

    if q_range is not None:
        if len(q_range) != 2:
            raise ValueError("q_range must be a tuple: (q_min, q_max)")
        q_min, q_max = q_range
        if q_min >= q_max:
            raise ValueError("q_range must satisfy q_min < q_max")
        valid_mask &= (q >= q_min) & (q <= q_max)

    if not np.any(valid_mask):
        raise ValueError("No valid Q points remain after applying q_range.")

    q_used = q[valid_mask]
    delta_fq_used_2d = delta_fq_2d[:, valid_mask]

    # ------------------------------------------------------------
    # Build modification/window function
    # ------------------------------------------------------------
    if window == "none":
        window_values = np.ones_like(q_used)

    elif window == "lorch":
        q_max_used = np.nanmax(q_used)
        if not np.isfinite(q_max_used) or q_max_used <= 0:
            raise ValueError("Maximum Q must be positive and finite for Lorch window.")

        x = np.pi * q_used / q_max_used
        window_values = np.ones_like(q_used)
        nonzero = x != 0
        window_values[nonzero] = np.sin(x[nonzero]) / x[nonzero]

    else:
        raise ValueError("window must be one of: 'none', 'lorch'")

    # ------------------------------------------------------------
    # Build r axis
    # ------------------------------------------------------------
    r = np.linspace(0.0, r_max, n_r)
    delta_gr_2d = np.full((delta_fq_used_2d.shape[0], n_r), np.nan, dtype=float)

    # ------------------------------------------------------------
    # Partially vectorized sine transform
    # ------------------------------------------------------------
    for i in range(delta_fq_used_2d.shape[0]):
        y = np.asarray(delta_fq_used_2d[i], dtype=float)

        finite_mask = np.isfinite(y) & np.isfinite(q_used)
        if np.sum(finite_mask) < 2:
            continue

        q_fit = q_used[finite_mask]
        y_fit = y[finite_mask]
        w_fit = window_values[finite_mask]

        fq_fit = y_fit * w_fit

        # Vectorized over all r values at once
        sin_qr = np.sin(np.outer(q_fit, r))  # shape: (n_q_fit, n_r)

        delta_gr_2d[i] = (2.0 / np.pi) * np.trapezoid(
            fq_fit[:, None] * sin_qr,
            q_fit,
            axis=0,
        )

    # ------------------------------------------------------------
    # Restore original dimensionality
    # ------------------------------------------------------------
    if input_was_1d:
        delta_gr = delta_gr_2d[0]
        delta_fq_used = delta_fq_used_2d[0]
    else:
        delta_gr = delta_gr_2d
        delta_fq_used = delta_fq_used_2d

        if not (0 <= profile_index < delta_fq_2d.shape[0]):
            raise ValueError(
                f"profile_index={profile_index} is out of bounds for {delta_fq_2d.shape[0]} profile(s)."
            )

    # ------------------------------------------------------------
    # Plot diagnostic example
    # ------------------------------------------------------------
    if plot:
        if input_was_1d:
            fq_plot = delta_fq_used_2d[0]
            gr_plot = delta_gr_2d[0]
            title_suffix = ""
        else:
            fq_plot = delta_fq_used_2d[profile_index]
            gr_plot = delta_gr_2d[profile_index]
            title_suffix = f" (Profile {profile_index})"

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].plot(q_used, fq_plot, label=r"$\Delta F(Q)$")
        axes[0].set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)")
        axes[0].set_ylabel(r"$\Delta F(Q)$")
        axes[0].set_title(f"Q-space Input{title_suffix}")
        axes[0].legend()

        axes[1].plot(r, gr_plot, label=r"$\Delta G(r)$")
        axes[1].set_xlabel(r"r ($\mathrm{\AA}$)")
        axes[1].set_ylabel(r"$\Delta G(r)$")
        axes[1].set_title(f"Real-space Transform{title_suffix}")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    if return_dict:
        return {
            "r": r,
            "delta_gr": delta_gr,
            "q_used": q_used,
            "delta_fq_used": delta_fq_used,
            "window_values": window_values,
            "q_range": q_range,
            "window": window,
            "input_was_1d": input_was_1d,
        }

    return r, delta_gr