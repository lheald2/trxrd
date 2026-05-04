import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf
from pathlib import Path

from globals import FIGSIZE, MASK_FILE, MASK_CENTER_X, MASK_CENTER_Y, MASK_RADIUS
from trxrd.io import _as_image_stack, _restore_image_dimensionality


def make_circular_mask(image_shape, center_xy, radius):
    """
    Create a circular boolean mask.

    Parameters
    ----------
    image_shape : tuple
        Image shape as (rows, cols).
    center_xy : tuple
        Circle center as (x0, y0) in pixel coordinates.
    radius : float
        Radius in pixels.

    Returns
    -------
    mask_bool : np.ndarray
        2D boolean mask where True indicates masked pixels.
    """
    rows, cols = image_shape
    y, x = np.indices((rows, cols))
    x0, y0 = center_xy

    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    mask_bool = r <= radius
    return mask_bool


def load_detector_mask(mask_path):
    """
    Load a detector mask from file.

    Parameters
    ----------
    mask_path : str or Path
        Path to mask image. Assumes pixels with value 0 are masked.

    Returns
    -------
    mask_bool : np.ndarray
        2D boolean mask where True indicates masked pixels.
    """
    mask = tf.imread(Path(mask_path))
    mask_bool = np.asarray(mask == 0, dtype=bool)
    return mask_bool


def build_combined_mask(
    image_shape,
    center_xy,
    radius,
    detector_mask=None,
    mask_path=None,
    plot=False,
    example_image=None,
    figsize=FIGSIZE,
    use_shared_color_scale=True,
):
    """
    Build a combined boolean mask from:
    - circular beam stop mask
    - detector mask

    Optionally plot the mask on an example image.

    Parameters
    ----------
    image_shape : tuple
        Shape of image as (rows, cols).
    center_xy : tuple
        Beam stop center as (x0, y0).
    radius : float
        Beam stop radius in pixels.
    detector_mask : np.ndarray, optional
        Preloaded 2D boolean detector mask where True = masked.
    mask_path : str or Path, optional
        Path to detector mask file. Used only if detector_mask is None.
    plot : bool, optional
        If True, plot the example image with mask overlay and masked image.
    example_image : np.ndarray, optional
        2D image used for visualization if plot=True.
    figsize : tuple, optional
        Figure size for plotting.
    use_shared_color_scale : bool, optional
        If True, both panels use the same color scale.

    Returns
    -------
    combined_mask : np.ndarray
        2D boolean mask where True indicates masked pixels.
    """
    beamstop_mask = make_circular_mask(
        image_shape=image_shape,
        center_xy=center_xy,
        radius=radius,
    )

    if detector_mask is None and mask_path is not None:
        detector_mask = load_detector_mask(mask_path)

    if detector_mask is None:
        combined_mask = beamstop_mask
    else:
        if detector_mask.shape != image_shape:
            raise ValueError(
                f"Detector mask shape {detector_mask.shape} does not match image shape {image_shape}."
            )
        combined_mask = beamstop_mask | detector_mask

    if plot:
        if example_image is None:
            raise ValueError("example_image must be provided when plot=True.")
        if example_image.shape != image_shape:
            raise ValueError(
                f"example_image shape {example_image.shape} does not match image shape {image_shape}."
            )

        original_image = np.array(example_image, dtype=float)

        # make a masked copy for display
        masked_image = original_image.copy()
        masked_image[combined_mask] = np.nan

        # safe log transform for visualization
        log_original = np.log1p(np.clip(original_image, a_min=0, a_max=None))
        log_masked = np.log1p(np.clip(masked_image, a_min=0, a_max=None))

        if use_shared_color_scale:
            finite_vals = log_original[np.isfinite(log_original)]
            vmin = np.nanmin(finite_vals)
            vmax = np.nanmax(finite_vals)
        else:
            vmin = None
            vmax = None

        _, axes = plt.subplots(1, 2, figsize=figsize)

        im0 = axes[0].imshow(log_original, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].contour(combined_mask, levels=[0.5], colors="white", linewidths=1.5)
        axes[0].scatter([center_xy[0]], [center_xy[1]], color="white", s=20, marker="x")
        axes[0].set_title("Log Image with Combined Mask")
        axes[0].set_xlabel("Pixel")
        axes[0].set_ylabel("Pixel")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(log_masked, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_title("Log Masked Image")
        axes[1].set_xlabel("Pixel")
        axes[1].set_ylabel("Pixel")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    return combined_mask


def apply_mask_from_bool(data_array, mask_bool):
    """
    Apply a precomputed 2D boolean mask to image data, replacing masked pixels with NaN.

    Parameters
    ----------
    data_array : np.ndarray
        Input image data, either:
        - 2D: (rows, cols)
        - 3D: (n_images, rows, cols)
    mask_bool : np.ndarray
        2D boolean mask where True indicates masked pixels.

    Returns
    -------
    masked_data : np.ndarray
        Float copy of input data with masked pixels set to NaN.
    """
    image_stack, input_was_2d = _as_image_stack(data_array, name="data_array")

    if image_stack.shape[1:] != mask_bool.shape:
        raise ValueError(
            f"Mask shape {mask_bool.shape} does not match image shape {image_stack.shape[1:]}."
        )

    masked_stack = image_stack.astype(float, copy=True)
    masked_stack[:, mask_bool] = np.nan

    return _restore_image_dimensionality(masked_stack, input_was_2d)

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

def apply_nan_mask(
    data_array,
    mask_path=MASK_FILE,
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
    n_images = image_stack.shape[0]

    if not (0 <= image_index < n_images):
        raise ValueError(
            f"image_index={image_index} is out of bounds for {n_images} image(s)."
        )

    mask_bool = load_detector_mask(mask_path)

    if image_stack.shape[1:] != mask_bool.shape:
        raise ValueError(
            f"Mask shape {mask_bool.shape} does not match image shape {image_stack.shape[1:]}."
        )

    masked_stack = apply_mask_from_bool(image_stack, mask_bool)

    if plot:
        original_image = image_stack[image_index]
        masked_image = masked_stack[image_index]

        log_original = np.log1p(original_image)
        log_masked = np.log1p(masked_image)

        if use_shared_color_scale:
            finite_vals = log_original[np.isfinite(log_original)]
            vmin = np.nanmin(finite_vals)
            vmax = np.nanmax(finite_vals)
        else:
            vmin = None
            vmax = None

        _, axes = plt.subplots(1, 2, figsize=figsize)

        im0 = axes[0].imshow(log_original, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].set_title("Log Original Image")
        axes[0].set_xlabel("Pixel")
        axes[0].set_ylabel("Pixel")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(log_masked, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_title("Log Masked Image")
        axes[1].set_xlabel("Pixel")
        axes[1].set_ylabel("Pixel")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    return _restore_image_dimensionality(masked_stack, input_was_2d)


def apply_beamstop_mask(
    data_array,
    center_xy=(MASK_CENTER_X, MASK_CENTER_Y),
    radius=MASK_RADIUS,
    plot=False,
    image_index=0,
    figsize=FIGSIZE,
    use_shared_color_scale=True,
):
    """
    Apply a circular beam stop mask to image data, replacing masked pixels with NaN.

    This function accepts either a single 2D image or a 3D image stack.
    Internally, the input is converted to a stack using `_as_image_stack`,
    the circular mask is broadcast across all images, and the original
    dimensionality is restored before returning.

    Parameters
    ----------
    data_array : np.ndarray
        Input image data, either:
        - 2D: (rows, cols)
        - 3D: (n_images, rows, cols)
    center_xy : tuple
        Beam center as (x0, y0) in pixel coordinates.
    radius : float
        Beam stop mask radius in pixels.
    plot : bool, optional
        If True, plot an example original image and masked image, with the
        beam stop mask overlaid on the original.
    image_index : int, optional
        Which image to plot if the input is a stack.
        If the input is a single 2D image, image_index must be 0.
    figsize : tuple, optional
        Figure size for the example plot.
    use_shared_color_scale : bool, optional
        If True, use the same color scale for the original and masked images.

    Returns
    -------
    masked_data : np.ndarray
        Float copy of input data with beam stop region set to NaN.
        Returns:
        - 2D array if input was 2D
        - 3D array if input was 3D

    Raises
    ------
    ValueError
        If the input dimensions are invalid, or if image_index is out of bounds.
    """
    image_stack, input_was_2d = _as_image_stack(data_array, name="data_array")
    n_images = image_stack.shape[0]
    image_shape = image_stack.shape[1:]

    if not (0 <= image_index < n_images):
        raise ValueError(
            f"image_index={image_index} is out of bounds for {n_images} image(s)."
        )

    mask_bool = make_circular_mask(
        image_shape=image_shape,
        center_xy=center_xy,
        radius=radius,
    )

    masked_stack = apply_mask_from_bool(image_stack, mask_bool)

    if plot:
        original_image = image_stack[image_index]
        masked_image = masked_stack[image_index]

        log_original = np.log1p(original_image)
        log_masked = np.log1p(masked_image)

        if use_shared_color_scale:
            finite_vals = log_original[np.isfinite(log_original)]
            vmin = np.nanmin(finite_vals)
            vmax = np.nanmax(finite_vals)
        else:
            vmin = None
            vmax = None

        _, axes = plt.subplots(1, 2, figsize=figsize)

        im0 = axes[0].imshow(log_original, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].contour(mask_bool, levels=[0.5], colors="white", linewidths=1.5)
        axes[0].scatter([center_xy[0]], [center_xy[1]], color="white", s=20, marker="x")
        axes[0].set_title("Log Image with Beam Stop Mask")
        axes[0].set_xlabel("Pixel")
        axes[0].set_ylabel("Pixel")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(log_masked, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_title("Log Masked Image")
        axes[1].set_xlabel("Pixel")
        axes[1].set_ylabel("Pixel")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    return _restore_image_dimensionality(masked_stack, input_was_2d)