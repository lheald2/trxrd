import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from functools import partial
from scipy.ndimage import gaussian_filter1d

from globals import FIGSIZE, STD_FACTOR, MAX_PROCESSORS
from trxrd.io import _as_image_stack, _restore_image_dimensionality, _get_counts


def _remove_xrays(
    image,
    mean_image,
    std_image,
    std_factor=STD_FACTOR,
    mask_bool=None,
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
        Threshold multiplier.
    mask_bool : np.ndarray, optional
        2D boolean mask where True indicates permanently masked pixels.

    Returns
    -------
    result : dict
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

    if mask_bool is not None:
        mask_bool = np.asarray(mask_bool, dtype=bool)
        if mask_bool.shape != image.shape:
            raise ValueError("mask_bool must have the same shape as image.")

    upper_threshold = mean_image + std_factor * std_image

    # Only search for hot pixels in valid, unmasked pixels
    bad_mask = image >= upper_threshold
    if mask_bool is not None:
        bad_mask = bad_mask & (~mask_bool)

    clean_image = image.copy()
    clean_image[bad_mask] = np.nan

    # Keep permanently masked pixels as NaN too
    if mask_bool is not None:
        clean_image[mask_bool] = np.nan

    n_removed = int(np.sum(bad_mask))
    valid_pixels = image.size if mask_bool is None else int(np.sum(~mask_bool))
    pct_removed = 100.0 * n_removed / valid_pixels if valid_pixels > 0 else np.nan

    return {
        "clean_image": clean_image,
        "bad_mask": bad_mask,
        "n_removed": n_removed,
        "pct_removed": pct_removed,
    }


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


def average_images_by_delay(
    data_dict,
    return_dict=True,
):
    """
    Group images by delay and compute the mean image for each delay.

    This function is intended to be used after filtering out bad images,
    for example after `remove_counts(...)`. It averages all remaining images
    that share the same delay value, producing one mean image per delay.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing at least:
        - "images" : np.ndarray of shape (n_images, rows, cols)
        - "delay" : np.ndarray of shape (n_images,)

        It may also contain other per-image metadata such as:
        - "file_names"
        - "image_number"
        - "sample_name"
        - "counts"
        - "fluence"
    return_dict : bool, optional
        If True, return a dictionary.
        If False, return tuple-style outputs.

    Returns
    -------
    result : dict or tuple
        If return_dict=True:
            {
                "images": np.ndarray of shape (n_delays, rows, cols),
                "std_images": np.ndarray of shape (n_delays, rows, cols),
                "delay": np.ndarray of shape (n_delays,),
                "counts_per_delay": np.ndarray of shape (n_delays,),
                "indices_by_delay": dict,
                "grouped_file_names": dict,
                "grouped_image_numbers": dict,
            }

        If return_dict=False:
            (unique_delays, mean_images, std_images, counts_per_delay)

    Raises
    ------
    ValueError
        If required keys are missing or shapes are inconsistent.

    Notes
    -----
    - The output "images" are the mean images for each delay.
    - "std_images" contains the standard deviation across images within each delay.
    - "grouped_file_names" and "grouped_image_numbers" keep traceability to the
      original files, but are stored as dictionaries because each delay may
      correspond to many input files.
    """
    required_keys = ["images", "delay"]
    for key in required_keys:
        if key not in data_dict:
            raise ValueError(f"data_dict is missing required key: '{key}'")

    images = np.asarray(data_dict["images"], dtype=float)
    delays = np.asarray(data_dict["delay"], dtype=float)

    if images.ndim != 3:
        raise ValueError("data_dict['images'] must be 3D with shape (n_images, rows, cols).")
    if delays.ndim != 1:
        raise ValueError("data_dict['delay'] must be 1D.")
    if images.shape[0] != len(delays):
        raise ValueError("Number of images must match number of delay values.")

    unique_delays = np.array(sorted(np.unique(delays)), dtype=float)

    mean_images = []
    std_images = []
    counts_per_delay = []
    indices_by_delay = {}
    grouped_file_names = {}
    grouped_image_numbers = {}

    for delay_val in unique_delays:
        idx = np.where(delays == delay_val)[0]
        indices_by_delay[delay_val] = idx
        counts_per_delay.append(len(idx))

        group = images[idx]
        mean_images.append(np.nanmean(group, axis=0))
        std_images.append(np.nanstd(group, axis=0))

        if "file_names" in data_dict:
            grouped_file_names[delay_val] = data_dict["file_names"][idx]
        if "image_number" in data_dict:
            grouped_image_numbers[delay_val] = data_dict["image_number"][idx]

    mean_images = np.asarray(mean_images, dtype=float)
    std_images = np.asarray(std_images, dtype=float)
    counts_per_delay = np.asarray(counts_per_delay, dtype=int)

    grouped_dict = {
        "images": mean_images,
        "std_images": std_images,
        "delay": unique_delays,
        "counts_per_delay": counts_per_delay,
        "indices_by_delay": indices_by_delay,
        "grouped_file_names": grouped_file_names,
        "grouped_image_numbers": grouped_image_numbers,
    }

    if return_dict:
        return grouped_dict

    return unique_delays, mean_images, std_images, counts_per_delay


def remove_xrays(
    data_array,
    std_factor=STD_FACTOR,
    plot=False,
    image_index=0,
    return_dict=True,
    mask_bool=None,
):
    """
    Remove hot pixels from one image or a stack of images using a threshold
    based on the stack mean and standard deviation.

    Parameters
    ----------
    data_array : np.ndarray
        Input image data, either:
        - 2D: (rows, cols)
        - 3D: (n_images, rows, cols)
    std_factor : float, optional
        Threshold multiplier used to identify hot pixels.
    plot : bool, optional
        If True, plot diagnostic images.
    image_index : int, optional
        Image index used for plotting when the input is a stack.
    return_dict : bool, optional
        If True, return a dictionary.
        If False, return only the cleaned data array.
    mask_bool : np.ndarray, optional
        2D boolean mask where True indicates permanently masked pixels.
        These pixels are excluded from the stack statistics and kept as NaN.

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

    working_stack = image_stack.astype(float, copy=True)

    if mask_bool is not None:
        mask_bool = np.asarray(mask_bool, dtype=bool)
        if working_stack.shape[1:] != mask_bool.shape:
            raise ValueError(
                f"Mask shape {mask_bool.shape} does not match image shape {working_stack.shape[1:]}."
            )
        working_stack[:, mask_bool] = np.nan

    mean_image = np.nanmean(working_stack, axis=0)
    std_image = np.nanstd(working_stack, axis=0)

    print(f"Removing hot pixels from {n_images} image(s)...")

    clean_list = []
    n_removed_list = []
    pct_removed_list = []

    for image in working_stack:
        result = _remove_xrays(
            image=image,
            mean_image=mean_image,
            std_image=std_image,
            std_factor=std_factor,
            mask_bool=mask_bool,
        )
        clean_list.append(result["clean_image"])
        n_removed_list.append(result["n_removed"])
        pct_removed_list.append(result["pct_removed"])

    clean_stack = np.stack(clean_list)
    n_removed = np.asarray(n_removed_list, dtype=int)
    pct_removed = np.asarray(pct_removed_list, dtype=float)

    clean_data = _restore_image_dimensionality(clean_stack, input_was_2d)

    if plot:
        _, axes = plt.subplots(1, 3, figsize=FIGSIZE)

        axes[0].plot(np.arange(n_images), pct_removed)
        axes[0].set_title("Percent Pixels Removed")
        axes[0].set_xlabel("Image Number")
        axes[0].set_ylabel("Percent")

        im1 = axes[1].imshow(working_stack[image_index], cmap="jet")
        axes[1].set_title("Original / Masked Image")
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
    mask_bool=None,
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

    working_stack = image_stack.astype(float, copy=True)

    if mask_bool is not None:
        mask_bool = np.asarray(mask_bool, dtype=bool)
        if working_stack.shape[1:] != mask_bool.shape:
            raise ValueError(
                f"Mask shape {mask_bool.shape} does not match image shape {working_stack.shape[1:]}."
            )
        working_stack[:, mask_bool] = np.nan

    mean_image = np.nanmean(working_stack, axis=0)
    std_image = np.nanstd(working_stack, axis=0)

    print(f"Removing hot pixels from {n_images} image(s)...")

    results = [None] * n_images

    worker = partial(
        _remove_xrays,
        mean_image=mean_image,
        std_image=std_image,
        std_factor=std_factor,
        mask_bool=mask_bool,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}

        for idx in range(n_images):
            future = executor.submit(worker, working_stack[idx])
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
        _, axes = plt.subplots(1, 3, figsize=FIGSIZE)

        axes[0].plot(np.arange(n_images), pct_removed)
        axes[0].set_title("Percent Pixels Removed")
        axes[0].set_xlabel("Image Number")
        axes[0].set_ylabel("Percent")

        im1 = axes[1].imshow(working_stack[image_index], cmap="jet")
        axes[1].set_title("Original / Masked Image")
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


def apply_gaussian_smoothing(matrix, sigma, dx=1.0, axis=0, mode="nearest"):
    """
    Apply Gaussian smoothing to a 2D matrix along one axis.

    Parameters
    ----------
    matrix : array-like
        Input 2D array.
    sigma : float
        Gaussian sigma in the same units as dx.
    dx : float, optional
        Step size along the smoothed axis.
    axis : int, optional
        Axis to smooth along (0 or 1).
    mode : str, optional
        Boundary handling mode for scipy.ndimage.gaussian_filter1d.

    Returns
    -------
    smoothed : np.ndarray
        Smoothed matrix.
    """
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")

    sigma_points = sigma / dx
    return gaussian_filter1d(matrix, sigma=sigma_points, axis=axis, mode=mode)


def make_reference_gr(
    grs,
    delays,
    reference_selector=None,
    return_dict=True,
):
    """
    Build a reference G(r) profile from selected curves, defaulting to
    all negative-delay curves.

    Parameters
    ----------
    grs : np.ndarray
        Array of shape (n_curves, n_r).
    delays : np.ndarray
        Array of shape (n_curves,).
    reference_selector : array-like, callable, or None
        If None, use delays < 0.
        If callable, should take delays and return boolean mask.
        If array-like, interpreted as boolean mask of shape (n_curves,).
    return_dict : bool, optional
        If True, return dictionary.

    Returns
    -------
    result : dict or tuple
        If return_dict=True:
            {
                "reference_gr": np.ndarray of shape (n_r,),
                "reference_std": np.ndarray of shape (n_r,),
                "reference_mask": np.ndarray of shape (n_curves,),
                "n_reference": int
            }

        If return_dict=False:
            (reference_gr, reference_std, reference_mask)
    """
    grs = np.asarray(grs, dtype=float)
    delays = np.asarray(delays, dtype=float)

    if grs.ndim != 2:
        raise ValueError("grs must have shape (n_curves, n_r)")
    if delays.ndim != 1:
        raise ValueError("delays must be 1D")
    if grs.shape[0] != len(delays):
        raise ValueError("grs and delays must have same number of curves")

    if reference_selector is None:
        reference_mask = delays < 0
    elif callable(reference_selector):
        reference_mask = np.asarray(reference_selector(delays), dtype=bool)
    else:
        reference_mask = np.asarray(reference_selector, dtype=bool)

    if reference_mask.shape != delays.shape:
        raise ValueError("reference_mask must have same shape as delays")

    if not np.any(reference_mask):
        raise ValueError("No reference curves selected")

    ref_group = grs[reference_mask]
    reference_gr = np.nanmean(ref_group, axis=0)
    reference_std = np.nanstd(ref_group, axis=0)

    if return_dict:
        return {
            "reference_gr": reference_gr,
            "reference_std": reference_std,
            "reference_mask": reference_mask,
            "n_reference": int(np.sum(reference_mask)),
        }

    return reference_gr, reference_std, reference_mask


def compute_delta_grs(
    grs,
    reference_gr,
    mode="subtract",
    delays=None,
    r=None,
    file_names=None,
    scan_number=None,
    sample_name=None,
    fluence=None,
    return_dict=True,
):
    """
    Compute delta G(r) curves relative to a reference G(r).

    Parameters
    ----------
    grs : np.ndarray
        Array of shape (..., n_r), usually (n_curves, n_r).
    reference_gr : np.ndarray
        Array of shape (n_r,).
    mode : {"subtract", "relative"}, optional
        "subtract" computes:
            delta_grs = grs - reference_gr
        "relative" computes:
            delta_grs = (grs - reference_gr) / reference_gr
    delays : np.ndarray or None, optional
        Delay values corresponding to each curve.
    r : np.ndarray or None, optional
        r axis.
    file_names : np.ndarray or None, optional
        File names corresponding to each curve.
    scan_number : np.ndarray or None, optional
        Scan numbers corresponding to each curve.
    sample_name : np.ndarray or None, optional
        Sample names corresponding to each curve.
    fluence : np.ndarray or None, optional
        Fluence values corresponding to each curve.
    return_dict : bool, optional
        If True, return dictionary.

    Returns
    -------
    result : dict or np.ndarray
        If return_dict=True, returns a dictionary containing delta_grs
        and any provided metadata.
    """
    grs = np.asarray(grs, dtype=float)
    reference_gr = np.asarray(reference_gr, dtype=float)

    if grs.shape[-1] != reference_gr.shape[0]:
        raise ValueError("Last dimension of grs must match reference_gr length")

    if mode == "subtract":
        delta_grs = grs - reference_gr
    elif mode == "relative":
        with np.errstate(divide="ignore", invalid="ignore"):
            delta_grs = (grs - reference_gr) / reference_gr
    else:
        raise ValueError("mode must be 'subtract' or 'relative'")

    if not return_dict:
        return delta_grs

    result = {
        "delta_grs": delta_grs,
        "reference_gr": reference_gr,
        "mode": mode,
    }

    if delays is not None:
        result["delay"] = np.asarray(delays, dtype=float)
    if r is not None:
        result["r"] = np.asarray(r, dtype=float)
    if file_names is not None:
        result["file_names"] = np.asarray(file_names)
    if scan_number is not None:
        result["scan_number"] = np.asarray(scan_number)
    if sample_name is not None:
        result["sample_name"] = np.asarray(sample_name)
    if fluence is not None:
        result["fluence"] = np.asarray(fluence)

    return result


def average_delta_grs_by_delay(
    data_dict,
    return_dict=True,
):
    """
    Group delta G(r) curves by delay and compute the mean delta G(r)
    for each delay.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing at least:
        - "delta_grs" : np.ndarray of shape (n_curves, n_r)
        - "delay" : np.ndarray of shape (n_curves,)

        It may also contain:
        - "r"
        - "file_names"
        - "scan_number"
        - "sample_name"
        - "fluence"

    return_dict : bool, optional
        If True, return a dictionary.
        If False, return tuple-style outputs.

    Returns
    -------
    result : dict or tuple
        If return_dict=True:
            {
                "r": np.ndarray of shape (n_r,),
                "delta_grs": np.ndarray of shape (n_delays, n_r),
                "std_delta_grs": np.ndarray of shape (n_delays, n_r),
                "delay": np.ndarray of shape (n_delays,),
                "counts_per_delay": np.ndarray of shape (n_delays,),
                "indices_by_delay": dict,
                "grouped_file_names": dict,
                "grouped_scan_numbers": dict,
            }

        If return_dict=False:
            (unique_delays, mean_delta_grs, std_delta_grs, counts_per_delay)

    Raises
    ------
    ValueError
        If required keys are missing or shapes are inconsistent.
    """
    required_keys = ["delta_grs", "delay"]
    for key in required_keys:
        if key not in data_dict:
            raise ValueError(f"data_dict is missing required key: '{key}'")

    delta_grs = np.asarray(data_dict["delta_grs"], dtype=float)
    delays = np.asarray(data_dict["delay"], dtype=float)

    if delta_grs.ndim != 2:
        raise ValueError(
            "data_dict['delta_grs'] must be 2D with shape (n_curves, n_r)."
        )
    if delays.ndim != 1:
        raise ValueError("data_dict['delay'] must be 1D.")
    if delta_grs.shape[0] != len(delays):
        raise ValueError("Number of delta G(r) curves must match number of delay values.")

    r = None
    if "r" in data_dict:
        r = np.asarray(data_dict["r"], dtype=float)
        if r.ndim != 1:
            raise ValueError("data_dict['r'] must be 1D.")
        if delta_grs.shape[1] != len(r):
            raise ValueError("Length of r must match second dimension of delta_grs.")

    unique_delays = np.array(sorted(np.unique(delays)), dtype=float)

    mean_delta_grs = []
    std_delta_grs = []
    counts_per_delay = []
    indices_by_delay = {}
    grouped_file_names = {}
    grouped_scan_numbers = {}

    for delay_val in unique_delays:
        idx = np.where(delays == delay_val)[0]
        indices_by_delay[delay_val] = idx
        counts_per_delay.append(len(idx))

        group = delta_grs[idx]
        mean_delta_grs.append(np.nanmean(group, axis=0))
        std_delta_grs.append(np.nanstd(group, axis=0))

        if "file_names" in data_dict:
            grouped_file_names[delay_val] = np.asarray(data_dict["file_names"])[idx]
        if "scan_number" in data_dict:
            grouped_scan_numbers[delay_val] = np.asarray(data_dict["scan_number"])[idx]

    mean_delta_grs = np.asarray(mean_delta_grs, dtype=float)
    std_delta_grs = np.asarray(std_delta_grs, dtype=float)
    counts_per_delay = np.asarray(counts_per_delay, dtype=int)

    grouped_dict = {
        "delta_grs": mean_delta_grs,
        "std_delta_grs": std_delta_grs,
        "delay": unique_delays,
        "counts_per_delay": counts_per_delay,
        "indices_by_delay": indices_by_delay,
        "grouped_file_names": grouped_file_names,
        "grouped_scan_numbers": grouped_scan_numbers,
    }

    if r is not None:
        grouped_dict["r"] = r

    if return_dict:
        return grouped_dict

    return unique_delays, mean_delta_grs, std_delta_grs, counts_per_delay
