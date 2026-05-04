import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from globals import (
    FIGSIZE,
    CENTER_X, CENTER_Y, DOWNSAMPLE,
    PIXEL1, PIXEL2, DISTANCE, WAVELENGTH,
    TILT_ANGLE, TILT_PLANE_ROTATION, ROT3,
    POLARIZATION_FACTOR, DARK, FLAT,
    UNIT, NAN_MIN, NAN_MAX, N_POINTS,
    MAX_PROCESSORS,
)
from trxrd.io import _as_image_stack, _restore_image_dimensionality
from trxrd.masking import build_pyfai_mask


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def yx_to_xy(center_yx):
    """Convert image-order center (y, x) -> pyFAI-order center (x, y)."""
    cy, cx = center_yx
    return (float(cx), float(cy))


def xy_to_yx(center_xy):
    """Convert pyFAI-order center (x, y) -> image-order center (y, x)."""
    cx, cy = center_xy
    return (float(cy), float(cx))


def _normalize_centers_xy(centers, n_images, use_average_center=False):
    """Normalize center input into an array of shape (n_images, 2) in (x, y) order."""
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


def normalize_centers(centers, n_images, use_average_center=False):
    """Backward-compatible wrapper for center normalization in (x, y) order."""
    return _normalize_centers_xy(centers, n_images, use_average_center=use_average_center)


# ---------------------------------------------------------------------------
# Detector geometry
# ---------------------------------------------------------------------------

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
    center_xy,
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
    center_xy : tuple or array-like
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
    x_center, y_center = center_xy

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


def custom_polarization_map_notebook(ai, image_shape, factor):
    """
    Reproduce the custom polarization correction from the earlier notebook
    using modern pyFAI center_array calls.

    Parameters
    ----------
    ai : pyFAI AzimuthalIntegrator
        Configured integrator.
    image_shape : tuple
        Shape of the image (rows, cols).
    factor : float
        Polarization factor in the same -1 to +1 style used by the notebook.

    Returns
    -------
    pol_map : np.ndarray
        2D multiplicative correction map.
    """
    tth = ai.center_array(shape=image_shape, unit="2th_rad")
    chi = ai.center_array(shape=image_shape, unit="chi_rad")

    f = (factor + 1.0) / 2.0

    denom = (
        f * (1.0 - (np.sin(tth) * np.sin(chi))**2) +
        (1.0 - f) * (1.0 - (np.sin(tth) * np.cos(chi))**2)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        pol_map = 1.0 / denom

    pol_map[~np.isfinite(pol_map)] = np.nan
    return pol_map


# ---------------------------------------------------------------------------
# Center finding
# ---------------------------------------------------------------------------

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
    """Score a radial profile by gradient energy."""
    if len(profile) < 3:
        return -np.inf
    grad = np.gradient(profile)
    return np.sum(grad ** 2)


def find_diffraction_center_from_guess_radial_fast(
    image,
    center_guess_yx,
    search_radius=20,
    mask=None,
    r_min=0,
    r_max=None,
    downsample=DOWNSAMPLE,
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
    center_guess_yx : tuple
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
    guess_cy, guess_cx = center_guess_yx

    if downsample > 1:
        img_work = img[::downsample, ::downsample]
        mask_work = None if mask is None else mask[::downsample, ::downsample]
        guess_cy_work = guess_cy / downsample
        guess_cx_work = guess_cx / downsample
        search_radius_work = max(1, int(np.ceil(search_radius / downsample)))
        r_min_work = r_min / downsample
        r_max_work = None if r_max is None else r_max / downsample
    else:
        img_work = img
        mask_work = mask
        guess_cy_work = guess_cy
        guess_cx_work = guess_cx
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
        int(np.round(guess_cy_work)) - search_radius_work,
        int(np.round(guess_cy_work)) + search_radius_work + 1,
    )
    cx_values = np.arange(
        int(np.round(guess_cx_work)) - search_radius_work,
        int(np.round(guess_cx_work)) + search_radius_work + 1,
    )

    score_map = np.full((len(cy_values), len(cx_values)), -np.inf, dtype=float)

    best_score = -np.inf
    best_center_yx_work = None
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
                best_center_yx_work = (cy, cx)
                best_r = r
                best_profile = profile

    if best_center_yx_work is None:
        raise RuntimeError("Could not determine a valid center.")

    best_cy_work, best_cx_work = best_center_yx_work

    if downsample > 1:
        best_center_yx = (best_cy_work * downsample, best_cx_work * downsample)
        best_r_full = best_r * downsample
        cy_values_full = cy_values * downsample
        cx_values_full = cx_values * downsample
    else:
        best_center_yx = best_center_yx_work
        best_r_full = best_r
        cy_values_full = cy_values
        cx_values_full = cx_values

    best_center_xy = yx_to_xy(best_center_yx)

    if plot:
        _, axes = plt.subplots(1, 3, figsize=figsize)

        img_plot = np.nan_to_num(img, nan=0.0)

        axes[0].imshow(img_plot, cmap="inferno")
        axes[0].plot(guess_cx, guess_cy, "co", label="Guess")
        axes[0].plot(best_center_xy[0], best_center_xy[1], "r+", ms=12, mew=2, label="Best")
        axes[0].set_title("Image with Center")
        axes[0].legend()

        im = axes[1].imshow(
            score_map,
            origin="lower",
            aspect="auto",
            extent=[cx_values_full[0], cx_values_full[-1], cy_values_full[0], cy_values_full[-1]],
            cmap="viridis",
        )
        axes[1].plot(guess_cx, guess_cy, "co", label="Guess")
        axes[1].plot(best_center_xy[0], best_center_xy[1], "r+", ms=12, mew=2, label="Best")
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
        "center_yx": np.asarray(best_center_yx, dtype=float),
        "center_xy": np.asarray(best_center_xy, dtype=float),
        "center_y": float(best_center_yx[0]),
        "center_x": float(best_center_yx[1]),
        "score": float(best_score),
        "score_map": score_map,
        "cy_values": cy_values_full,
        "cx_values": cx_values_full,
        "r": best_r_full,
        "radial_profile": best_profile,
    }


def _center_worker(
    idx,
    image,
    center_guess_yx,
    search_radius,
    mask,
    r_min,
    r_max,
    downsample,
    intensity_threshold,
    top_percentile,
):
    """Worker for one image, suitable for parallel execution."""
    result = find_diffraction_center_from_guess_radial_fast(
        image=image,
        center_guess_yx=center_guess_yx,
        search_radius=search_radius,
        mask=mask,
        r_min=r_min,
        r_max=r_max,
        downsample=downsample,
        intensity_threshold=intensity_threshold,
        top_percentile=top_percentile,
        plot=False,
    )

    center_yx = np.asarray(result["center_yx"], dtype=float)
    center_xy = np.asarray(result["center_xy"], dtype=float)

    return {
        "index": idx,
        "center_yx": center_yx,
        "center_xy": center_xy,
        "center_y": float(center_yx[0]),
        "center_x": float(center_yx[1]),
        "score": float(result["score"]),
        "full_result": result,
    }


def find_centers_in_stack_radial_parallel(
    data_array,
    center_guess_yx=(CENTER_Y, CENTER_X),
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
    """
    Find diffraction centers for one image or a stack of images in parallel.

    Internal center convention is (y, x). Returned dict includes both
    centers_yx and centers_xy.
    """
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
                center_guess_yx=center_guess_yx,
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

    centers_yx = np.vstack([d["center_yx"] for d in results_list]).astype(float)
    centers_xy = np.vstack([d["center_xy"] for d in results_list]).astype(float)

    center_y = centers_yx[:, 0]
    center_x = centers_yx[:, 1]
    score = np.array([d["score"] for d in results_list], dtype=float)
    image_index = np.arange(n_images)
    per_image_results = [d["full_result"] for d in results_list]

    if plot_example:
        if not (0 <= example_index < n_images):
            raise ValueError("example_index is out of bounds.")

        example_result = per_image_results[example_index]
        img = image_stack[example_index].astype(float)
        img_plot = np.nan_to_num(img, nan=0.0)

        _, axes = plt.subplots(1, 3, figsize=figsize_example)

        axes[0].imshow(img_plot, cmap="inferno")
        axes[0].plot(center_guess_yx[1], center_guess_yx[0], "co", label="Fixed guess")
        axes[0].plot(centers_xy[example_index, 0], centers_xy[example_index, 1], "r+", ms=12, mew=2, label="Best")
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
        axes[1].plot(center_guess_yx[1], center_guess_yx[0], "co", label="Fixed guess")
        axes[1].plot(centers_xy[example_index, 0], centers_xy[example_index, 1], "r+", ms=12, mew=2, label="Best")
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

        _, axes = plt.subplots(2, 1, figsize=figsize_trend, sharex=True)
        axes[0].plot(xvals, center_x, "o-")
        axes[0].set_ylabel("Center x (pixels)")
        axes[0].set_title("Center Position vs Image")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(xvals, center_y, "o-")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel("Center y (pixels)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return {
        "center_y": center_y,
        "center_x": center_x,
        "centers_yx": centers_yx,
        "centers_xy": centers_xy,
        "score": score,
        "image_index": image_index,
        "per_image_results": per_image_results,
        "input_was_2d": input_was_2d,
    }


# ---------------------------------------------------------------------------
# Azimuthal integration
# ---------------------------------------------------------------------------

def _azimuthal_worker(
    idx,
    image,
    center_xy,
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
    use_custom_polarization=False,
    integration_function="integrate1d",
    correct_solid_angle=False,
):
    """
    Worker function for azimuthal integration of a single image.

    Parameters
    ----------
    idx : int
        Image index (used for ordering results).
    image : np.ndarray
        2D diffraction image.
    center_xy : array-like
        (x_center, y_center) for this image.
    npt : int
        Number of radial bins.
    unit : str
        Radial unit for integration.
    radial_range : tuple or None
        Radial range passed directly to pyFAI integration.
    nan_radial_range : tuple or None
        Range to keep in the returned profile. Values outside this range
        are replaced with NaN after integration.
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
    use_custom_polarization : bool, optional
        If True, apply notebook-style custom polarization as a 2D map
        before integration, then disable pyFAI built-in polarization.
    integration_function : {"integrate1d", "integrate1d_ng"}, optional
        Which pyFAI 1D integrator to use.
    correct_solid_angle : bool, optional
        Whether to apply pyFAI solid-angle correction.

    Returns
    -------
    dict
        Dictionary containing index, radial, intensity, success, and error.
    """
    try:
        ai = make_azimuthal_integrator(
            center_xy=center_xy,
            pixel1=pixel1,
            pixel2=pixel2,
            distance=distance,
            wavelength=wavelength,
            tilt_angle=tilt_angle,
            tilt_plane_rotation=tilt_plane_rotation,
            rot3=rot3,
        )

        image_mask, clean_image = build_pyfai_mask(image, mask=mask)

        if use_custom_polarization and polarization_factor is not None:
            pol_map = custom_polarization_map_notebook(
                ai,
                clean_image.shape,
                factor=polarization_factor,
            )
            image_for_integration = clean_image * pol_map
            pyfai_pol = None
        else:
            image_for_integration = clean_image
            pyfai_pol = polarization_factor

        if integration_function == "integrate1d":
            radial, intensity = ai.integrate1d(
                image_for_integration,
                npt=npt,
                unit=unit,
                radial_range=radial_range,
                azimuth_range=azimuth_range,
                mask=image_mask,
                dark=dark,
                flat=flat,
                polarization_factor=pyfai_pol,
                correctSolidAngle=correct_solid_angle,
                method=method,
            )
        elif integration_function == "integrate1d_ng":
            res = ai.integrate1d_ng(
                image_for_integration,
                npt=npt,
                unit=unit,
                radial_range=radial_range,
                azimuth_range=azimuth_range,
                mask=image_mask,
                dark=dark,
                flat=flat,
                polarization_factor=pyfai_pol,
                correctSolidAngle=correct_solid_angle,
                method=method,
            )

            if hasattr(res, "radial") and hasattr(res, "intensity"):
                radial = res.radial
                intensity = res.intensity
            else:
                radial, intensity = res
        else:
            raise ValueError(
                "integration_function must be 'integrate1d' or 'integrate1d_ng'"
            )

        radial = np.asarray(radial, dtype=float)
        intensity = np.asarray(intensity, dtype=float)

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
    centers_xy,
    use_average_center=False,
    npt=N_POINTS,
    unit=UNIT,
    radial_range=None,
    nan_radial_range=(NAN_MIN, NAN_MAX),
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
    use_custom_polarization=False,
    integration_function="integrate1d",
    correct_solid_angle=False,
    **kwargs,
):
    """
    Compute azimuthal averages for one image or a stack of images using pyFAI.

    Parameters
    ----------
    images : np.ndarray
        2D image or 3D image stack.
    centers_xy : tuple or np.ndarray
        Center(s) in (x, y) pixel coordinates.
    use_average_center : bool, optional
        If True and centers_xy are provided per-image, average them and use one
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
        Print progress every this many completed images.

    Returns
    -------
    dict or tuple
        If return_dict=True, returns a dict with keys: radial, profiles,
        centers_used_xy, centers_used_yx, success, geometry, unit,
        input_was_2d, radial_range, nan_radial_range.
        Otherwise returns (radial, profile) for 2D input or
        (radial, profiles) for 3D input.
    """
    if "mask" in kwargs and integration_mask is None:
        integration_mask = kwargs.pop("mask")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

    image_stack, input_was_2d = _as_image_stack(images, name="images")
    n_images = image_stack.shape[0]

    centers_used_xy = _normalize_centers_xy(
        centers_xy,
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
                center_xy=centers_used_xy[idx],
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
                use_custom_polarization=use_custom_polarization,
                integration_function=integration_function,
                correct_solid_angle=correct_solid_angle,
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
        "use_custom_polarization": use_custom_polarization,
        "integration_function": integration_function,
        "correct_solid_angle": correct_solid_angle,
    }

    if return_dict:
        return {
            "radial": radial_out,
            "profiles": profiles,
            "centers_used_xy": centers_used_xy,
            "centers_used_yx": np.column_stack((centers_used_xy[:, 1], centers_used_xy[:, 0])),
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


def get_polar_map(
    ai,
    image,
    mask=None,
    pol=None,
    npt_rad=500,
    npt_azim=360,
    q_unit="q_A^-1",
    correct_solid_angle=True,
    method=("bbox", "csr", "cython"),
):
    """
    Compute I(q, chi) using pyFAI integrate2d.

    Returns
    -------
    I_qchi : ndarray, shape (n_chi, n_q)
        2D integrated intensity.
    q : ndarray
        Radial coordinate.
    chi : ndarray
        Azimuthal coordinate.
    """
    res = ai.integrate2d(
        image,
        npt_rad=npt_rad,
        npt_azim=npt_azim,
        mask=mask,
        polarization_factor=pol,
        correctSolidAngle=correct_solid_angle,
        unit=q_unit,
        method=method,
    )

    if hasattr(res, "intensity"):
        I_qchi = res.intensity
        q = res.radial
        chi = res.azimuthal
    else:
        I_qchi, q, chi = res

    I_qchi = np.asarray(I_qchi)
    q = np.asarray(q)
    chi = np.asarray(chi)

    if I_qchi.shape == (len(q), len(chi)):
        I_qchi = I_qchi.T

    return I_qchi, q, chi


def azimuthal_anisotropy(I_qchi):
    """
    Compute azimuthal std and relative std at each q.
    Expects I_qchi shape = (n_chi, n_q).
    """
    mean_q = np.nanmean(I_qchi, axis=0)
    std_q = np.nanstd(I_qchi, axis=0)
    rel_std_q = std_q / mean_q
    return mean_q, std_q, rel_std_q
