import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

from globals import (
    FIGSIZE,
    CENTER_X, CENTER_Y, DOWNSAMPLE,
    UNIT, NAN_MIN, NAN_MAX,
    POLARIZATION_FACTOR, DARK, FLAT,
    NORM_MIN, NORM_MAX,
    LAM_VAL, P_VAL,
    MAX_PROCESSORS,
    PONI_FILE,
)
from trxrd.io import _as_image_stack
from trxrd.integration import (
    _normalize_centers_xy,
    find_diffraction_center_from_guess_radial_fast,
    find_centers_in_stack_radial_parallel,
    azimuthal_average_pyfai,
)


def compute_background_azimuthal_average(
    background_input,
    centers_xy=None,
    poni_path=PONI_FILE,
    center_guess_yx=(CENTER_Y, CENTER_X),
    compute_center_if_missing=True,
    center_from="mean",
    search_radius=20,
    center_mask=None,
    r_min=0,
    r_max=1400,
    downsample=DOWNSAMPLE,
    intensity_threshold=None,
    top_percentile=60,
    npt=5000,
    radial_range=None,
    nan_radial_range=(NAN_MIN, NAN_MAX),
    azimuth_range=None,
    integration_mask=None,
    unit=UNIT,
    method=("bbox", "csr", "cython"),
    polarization_factor=POLARIZATION_FACTOR,
    dark=DARK,
    flat=FLAT,
    use_custom_polarization=False,
    integration_function="integrate1d",
    correct_solid_angle=False,
    error_mode="raise",
    max_workers=None,
    progress_interval=100,
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
        - dict containing "background_stack" or "background_mean"
    centers_xy : array-like or None, optional
        Beam center(s) in (x, y) pixel coordinates.
        If None and poni_path is provided, the center is read from the PONI file.
        If None and poni_path is None, centers are computed if compute_center_if_missing=True.
    poni_path : path-like or None, optional
        Path to a .poni file produced by pyFAI-calib2. When supplied, the PONI
        geometry (distance, wavelength, tilt, center) is used and the manual
        detector parameters in azimuthal_average_pyfai are ignored.
        Defaults to PONI_FILE from globals.
    center_guess_yx : tuple, optional
        Initial guess for center finding as (y, x).
    compute_center_if_missing : bool, optional
        If True and centers_xy is None, automatically determine centers.
    center_from : {"mean", "each"}, optional
        Strategy for automatic center determination.
    search_radius : int, optional
        Pixel radius around the center_guess to search for the optimal center.
    center_mask : np.ndarray or None, optional
        Boolean mask used during center finding (True = excluded pixel).
    r_min : int, optional
        Minimum radius (in pixels) used when evaluating radial profiles.
    r_max : int or None, optional
        Maximum radius (in pixels) used when evaluating radial profiles.
    downsample : int, optional
        Downsampling factor for center finding.
    intensity_threshold : float or None, optional
        Minimum intensity threshold for selecting pixels during center finding.
    top_percentile : float or None, optional
        Use only pixels above this percentile for center finding.
    npt : int, optional
        Number of radial bins for azimuthal integration.
    radial_range : tuple or None, optional
        Radial range passed directly to pyFAI integration.
    nan_radial_range : tuple or None, optional
        Radial range to keep after integration; values outside are set to NaN.
    azimuth_range : tuple or None, optional
        Azimuthal integration range in degrees.
    integration_mask : np.ndarray or None, optional
        Boolean mask applied during azimuthal integration (True = excluded).
    unit : str, optional
        Radial unit for output.
    method : tuple or str, optional
        Integration method passed to pyFAI.
    polarization_factor : float or None, optional
        Polarization correction factor.
    dark : np.ndarray or None, optional
        Dark current image for correction.
    flat : np.ndarray or None, optional
        Flat-field correction image.
    use_custom_polarization : bool, optional
        If True, apply the notebook-style custom polarization correction.
    integration_function : {"integrate1d", "integrate1d_ng"}, optional
        Which pyFAI 1D integration function to use.
    correct_solid_angle : bool, optional
        Whether to apply pyFAI solid-angle correction during integration.
    error_mode : {"raise", "warn", "ignore"}, optional
        Error handling mode passed to azimuthal averaging.
    max_workers : int or None, optional
        Number of worker threads for azimuthal averaging.
    progress_interval : int, optional
        Print progress every this many completed images.
    plot : bool, optional
        If True, display an example background image and its azimuthal profile.
    image_index : int, optional
        Index of image to use for plotting if multiple images are present.
    figsize : tuple, optional
        Figure size for plotting.
    return_dict : bool, optional
        If True, return results as a dictionary.

    Returns
    -------
    result : dict or tuple
        If return_dict=True:
            {
                "radial": np.ndarray,
                "background_profiles": np.ndarray,
                "background_profile_mean": np.ndarray,
                "background_profile_std": np.ndarray,
                "background_images_used": np.ndarray,
                "centers_used_xy": np.ndarray,
                "centers_used_yx": np.ndarray,
                "center_result": dict or None,
                "pyfai_result": dict,
                "input_was_2d": bool,
                "radial_range": tuple or None,
                "nan_radial_range": tuple or None,
            }
        If return_dict=False:
            (radial, background_profiles, background_profile_mean)
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

    background_images, input_was_2d = _as_image_stack(
        background_images,
        name="background_input",
    )
    n_bg = background_images.shape[0]

    if not (0 <= image_index < n_bg):
        raise ValueError(
            f"image_index={image_index} is out of bounds for {n_bg} background image(s)."
        )

    center_result = None

    if centers_xy is not None:
        centers_xy_array = _normalize_centers_xy(centers_xy, n_bg, use_average_center=False)
    elif poni_path is not None:
        centers_xy_array = None  # azimuthal_average_pyfai reads the center from the poni file
    else:
        if not compute_center_if_missing:
            raise ValueError("centers_xy is None, poni_path is None, and compute_center_if_missing=False.")

        if center_from == "mean":
            mean_bg = np.nanmean(background_images, axis=0)

            center_result = find_diffraction_center_from_guess_radial_fast(
                image=mean_bg,
                center_guess_yx=center_guess_yx,
                search_radius=search_radius,
                mask=center_mask,
                r_min=r_min,
                r_max=r_max,
                downsample=downsample,
                intensity_threshold=intensity_threshold,
                top_percentile=top_percentile,
                plot=False,
            )

            one_center_xy = np.asarray(center_result["center_xy"], dtype=float)
            centers_xy_array = np.tile(one_center_xy, (n_bg, 1))

        elif center_from == "each":
            center_result = find_centers_in_stack_radial_parallel(
                data_array=background_images,
                center_guess_yx=center_guess_yx,
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
            centers_xy_array = np.asarray(center_result["centers_xy"], dtype=float)

        else:
            raise ValueError("center_from must be 'mean' or 'each'")

    pyfai_result = azimuthal_average_pyfai(
        images=background_images,
        centers_xy=centers_xy_array,
        poni_path=poni_path,
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
        use_custom_polarization=use_custom_polarization,
        integration_function=integration_function,
        correct_solid_angle=correct_solid_angle,
        return_dict=True,
        error_mode=error_mode,
        max_workers=max_workers,
        progress_interval=progress_interval,
    )

    if centers_xy_array is None:
        centers_xy_array = pyfai_result["centers_used_xy"]

    radial = pyfai_result["radial"]
    background_profiles = pyfai_result["profiles"]
    background_profile_mean = np.nanmean(background_profiles, axis=0)
    background_profile_std = np.nanstd(background_profiles, axis=0)

    if plot:
        _, axes = plt.subplots(1, 2, figsize=figsize)

        im = axes[0].imshow(background_images[image_index], cmap="jet")
        cx, cy = centers_xy_array[image_index]
        axes[0].plot(cx, cy, "wo", ms=8, mec="k")
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
            "centers_used_xy": centers_xy_array,
            "centers_used_yx": np.column_stack((centers_xy_array[:, 1], centers_xy_array[:, 0])),
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

    Parameters
    ----------
    radial : np.ndarray
        1D radial axis of shape (n_q,).
    profiles : np.ndarray
        Input 1D profiles, either 1D (n_q,) or 2D (n_profiles, n_q).
    background_profile : np.ndarray
        1D background profile of shape (n_q,).
    norm_range : tuple
        Radial range (r_min, r_max) used to determine the background scale factor.
    mode : {"mean", "sum", "median", "max"}, optional
        Statistic used when scale_method="ratio".
    scale_method : {"ratio", "least_squares"}, optional
        Method used to compute the scale factor for each profile.
    plot : bool, optional
        If True, plot selected original, scaled background, and corrected profiles.
    plot_scale_factors : bool, optional
        If True, plot the background scale factor versus profile index.
    plot_indices : None, int, or sequence of int, optional
        Which profiles to plot if plot=True. If None, plots the first profile.
    figsize : tuple, optional
        Figure size for plotting.
    alpha : float, optional
        Line transparency for profile plots.
    return_dict : bool, optional
        If True, return a dictionary. If False, return a tuple.

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

    scaled_background_profiles = scale_factors[:, None] * background_profile[None, :]
    corrected_profiles_2d = profiles_2d - scaled_background_profiles

    if plot_scale_factors:
        plt.figure(figsize=figsize)
        plt.plot(np.arange(len(scale_factors)), scale_factors, lw=1.5)
        # plt.axhline(1.0, color="k", linestyle="--", alpha=0.6)
        plt.xlabel("Profile Index")
        plt.ylabel("Background Scale Factor")
        plt.title("Background Scale Factor vs Profile Index")
        plt.tight_layout()
        plt.show()

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

        _, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)

        for idx in plot_indices:
            axes[0].plot(radial, profiles_2d[idx], alpha=alpha, label=f"Profile {idx}")
        axes[0].axvspan(r_min, r_max, color="gray", alpha=0.2)
        axes[0].set_title("Original Profiles")
        axes[0].set_xlabel("Radial coordinate")
        axes[0].set_ylabel("Intensity")
        axes[0].legend()

        for idx in plot_indices:
            axes[1].plot(radial, scaled_background_profiles[idx], alpha=alpha, label=f"Scaled BG {idx}")
        axes[1].axvspan(r_min, r_max, color="gray", alpha=0.2)
        axes[1].set_title("Scaled Background Profiles")
        axes[1].set_xlabel("Radial coordinate")
        axes[1].legend()

        for idx in plot_indices:
            axes[2].plot(radial, corrected_profiles_2d[idx], alpha=alpha, label=f"Corrected {idx}")
        axes[2].axvspan(r_min, r_max, color="gray", alpha=0.2)
        axes[2].set_title("Background-Subtracted Profiles")
        axes[2].set_xlabel("Radial coordinate")
        axes[2].legend()

        plt.tight_layout()
        plt.show()

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
        Precomputed normalized profiles. Required if show_normalized=True.
    plot_indices : None, int, or sequence of int, optional
        Which profiles to plot. If None, plots the first profile.
    show_normalized : bool, optional
        If True, also plot normalized profiles in a second panel.
    figsize : tuple, optional
        Figure size.
    alpha : float, optional
        Line transparency.
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
        _, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
        axes = np.atleast_1d(axes)

        panel_data = [
            (axes[0], profiles, "Original Profiles"),
            (axes[1], normalized_profiles, "Normalized Profiles"),
        ]
    else:
        _, ax = plt.subplots(1, 1, figsize=figsize)
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
    norm_range=(NORM_MIN, NORM_MAX),
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
        Statistic used to compute the normalization factor inside norm_range.
    return_dict : bool, optional
        If True, return a dictionary. If False, return tuple.
    plot : bool, optional
        If True, plot the normalization window on selected profiles.
    plot_indices : None, int, or sequence of int, optional
        Which profiles to plot.
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
        _, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(factors))
        ax.plot(x, factors, lw=1.5, label="Normalization factor")

        negative_mask = factors < 0
        if np.any(negative_mask):
            ax.scatter(x[negative_mask], factors[negative_mask], s=18, label="Negative factors", zorder=3)

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
    """Internal helper: compute ALS baseline for a single 1D array."""
    y = np.asarray(y, dtype=float)
    n = len(y)

    finite_mask = np.isfinite(y)
    if np.sum(finite_mask) < 3:
        return np.full_like(y, np.nan)

    if not np.all(finite_mask):
        x = np.arange(n)
        y_fit = np.interp(x, x[finite_mask], y[finite_mask])
    else:
        y_fit = y.copy()

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
    lam=LAM_VAL,
    p=P_VAL,
    niter=10,
    plot=False,
    profile_index=0,
    x_vals=None,
    figsize=FIGSIZE,
    return_dict=True,
):
    """
    Subtract a smooth baseline using asymmetric least squares (ALS).

    Parameters
    ----------
    data_array : np.ndarray
        Input data: 1D array (n_x,) or 2D array (n_profiles, n_x).
    lam : float, optional
        Smoothness parameter. Larger values → smoother baseline. Typical: 1e4–1e8.
    p : float, optional
        Asymmetry parameter (0 < p < 1). Smaller → baseline stays below peaks.
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
                "input_was_1d": bool,
            }
        If return_dict=False:
            (corrected_data, baselines)
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

    baselines = np.asarray(
        [_als_baseline_1d(data_stack[i], lam, p, niter) for i in range(n_profiles)],
        dtype=float,
    )
    corrected_stack = data_stack - baselines

    if input_was_1d:
        corrected_out = corrected_stack[0]
        baselines_out = baselines[0]
    else:
        corrected_out = corrected_stack
        baselines_out = baselines

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

        _, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)

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


def apply_polynomial_baseline(
    q,
    fq,
    q_fit_range=None,
    poly_order=2,
    smooth_window=51,
    smooth_polyorder=3,
    plot=False,
    profile_index=0,
    figsize=(12, 4),
    return_dict=True,
):
    """
    Remove a slowly varying polynomial baseline from F(Q).

    The polynomial is fit to a heavily smoothed version of F(Q), so that
    broad baseline drift is modeled without strongly fitting the real
    oscillatory structure.

    Parameters
    ----------
    q : np.ndarray
        1D Q axis of shape (n_q,).
    fq : np.ndarray
        F(Q), either 1D (n_q,) or 2D (n_profiles, n_q).
    q_fit_range : tuple or None, optional
        (q_min, q_max) range used for baseline fitting.
        If None, use all finite Q points.
    poly_order : int, optional
        Polynomial order for the baseline fit. Recommended: 1 or 2.
    smooth_window : int, optional
        Window length for Savitzky-Golay smoothing. Must be odd.
    smooth_polyorder : int, optional
        Polynomial order used in Savitzky-Golay smoothing.
    plot : bool, optional
        If True, plot one example profile.
    profile_index : int, optional
        Which profile to plot if fq is 2D.
    figsize : tuple, optional
        Figure size for plotting.
    return_dict : bool, optional
        If True, return a dictionary. If False, return corrected F(Q) only.

    Returns
    -------
    result : dict or np.ndarray
        If return_dict=True:
            {
                "q": q,
                "fq_corrected": np.ndarray,
                "baseline": np.ndarray,
                "fq_smoothed": np.ndarray,
                "coefficients": list or np.ndarray,
                "q_fit_range": tuple or None,
                "poly_order": int,
                "input_was_1d": bool,
            }
        If return_dict=False:
            fq_corrected
    """
    q = np.asarray(q, dtype=float)
    fq = np.asarray(fq, dtype=float)

    if q.ndim != 1:
        raise ValueError("q must be 1D.")

    if fq.ndim == 1:
        fq_2d = fq[None, :]
        input_was_1d = True
    elif fq.ndim == 2:
        fq_2d = fq
        input_was_1d = False
    else:
        raise ValueError("fq must be 1D or 2D.")

    if fq_2d.shape[1] != q.shape[0]:
        raise ValueError("fq.shape[-1] must match len(q).")

    if poly_order < 0:
        raise ValueError("poly_order must be >= 0.")

    if smooth_window % 2 == 0:
        smooth_window += 1

    if smooth_window <= smooth_polyorder:
        raise ValueError("smooth_window must be greater than smooth_polyorder.")

    fit_mask = np.isfinite(q)
    if q_fit_range is not None:
        if len(q_fit_range) != 2:
            raise ValueError("q_fit_range must be a tuple: (q_min, q_max)")
        q_min, q_max = q_fit_range
        if q_min >= q_max:
            raise ValueError("q_fit_range must satisfy q_min < q_max")
        fit_mask &= (q >= q_min) & (q <= q_max)

    if np.sum(fit_mask) < poly_order + 2:
        raise ValueError("Not enough points to fit the requested polynomial baseline.")

    fq_corrected_2d = np.full_like(fq_2d, np.nan, dtype=float)
    baseline_2d = np.full_like(fq_2d, np.nan, dtype=float)
    fq_smoothed_2d = np.full_like(fq_2d, np.nan, dtype=float)
    coefficients = []

    for i in range(fq_2d.shape[0]):
        y = np.asarray(fq_2d[i], dtype=float)
        finite = np.isfinite(q) & np.isfinite(y)

        if np.sum(finite) < max(smooth_window, poly_order + 2):
            coefficients.append(None)
            continue

        y_interp = y.copy()
        if not np.all(finite):
            y_interp[~finite] = np.interp(q[~finite], q[finite], y[finite])

        y_smooth = savgol_filter(y_interp, window_length=smooth_window, polyorder=smooth_polyorder)
        fq_smoothed_2d[i] = y_smooth

        local_fit_mask = fit_mask & np.isfinite(y_smooth)
        x_fit = q[local_fit_mask]
        y_fit = y_smooth[local_fit_mask]

        coeff = np.polyfit(x_fit, y_fit, deg=poly_order)
        coefficients.append(coeff)

        baseline = np.polyval(coeff, q)
        baseline_2d[i] = baseline
        fq_corrected_2d[i] = y - baseline

    if input_was_1d:
        fq_corrected = fq_corrected_2d[0]
        baseline = baseline_2d[0]
        fq_smoothed = fq_smoothed_2d[0]
        coefficients_out = coefficients[0]
    else:
        fq_corrected = fq_corrected_2d
        baseline = baseline_2d
        fq_smoothed = fq_smoothed_2d
        coefficients_out = coefficients

        if not (0 <= profile_index < fq_2d.shape[0]):
            raise ValueError(
                f"profile_index={profile_index} is out of bounds for {fq_2d.shape[0]} profile(s)."
            )

    if plot:
        idx = 0 if input_was_1d else profile_index

        _, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)

        axes[0].plot(q, fq_2d[idx], label="Original F(Q)", alpha=0.7)
        axes[0].plot(q, fq_smoothed_2d[idx], label="Smoothed F(Q)", linewidth=2)
        axes[0].plot(q, baseline_2d[idx], label=f"Polynomial baseline (order {poly_order})", linewidth=2)
        if q_fit_range is not None:
            axes[0].axvspan(q_fit_range[0], q_fit_range[1], alpha=0.15, label="Fit range")
        axes[0].axhline(0, linestyle="--")
        axes[0].set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)")
        axes[0].set_ylabel("F(Q)")
        axes[0].set_title("Baseline Fit")
        axes[0].legend()

        axes[1].plot(q, fq_corrected_2d[idx], label="Baseline-corrected F(Q)")
        axes[1].axhline(0, linestyle="--")
        axes[1].set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)")
        axes[1].set_ylabel("F(Q)")
        axes[1].set_title("Corrected F(Q)")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    if return_dict:
        return {
            "q": q,
            "fq_corrected": fq_corrected,
            "baseline": baseline,
            "fq_smoothed": fq_smoothed,
            "coefficients": coefficients_out,
            "q_fit_range": q_fit_range,
            "poly_order": poly_order,
            "input_was_1d": input_was_1d,
        }

    return fq_corrected
