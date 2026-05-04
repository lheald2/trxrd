import re
from pathlib import Path

import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt

from globals import FIGSIZE, DELAY_SIGN, SCAN_NAME


# ---------------------------------------------------------------------------
# Filename patterns
# ---------------------------------------------------------------------------

FILENAME_PATTERNS = {
    "delay_scan": re.compile(
        r"^(?P<sample_name>[A-Za-z0-9_]+)-"
        r"(?P<fluence>[-+]?\d*\.?\d+)fs"
        r"hw(?P<delay>[-+]?\d*\.?\d+(?:e[-+]?\d+)?)"
        r"delay(?P<image_number>\d+)\.tif$",
        re.IGNORECASE,
    ),
    "theta_samz": re.compile(
        r"^(?P<sample_name>[A-Za-z]+[A-Za-z0-9]*)"
        r"(?P<scan_id>M\d+)_"
        r"(?P<degree>[-+]?\d*\.?\d+)"
        r"-deg_theta"
        r"(?P<theta>[-+]?\d*\.?\d+)"
        r"samz(?P<samz>\d+)_"
        r"(?P<image_number>\d+)\.tif$",
        re.IGNORECASE,
    ),
}

_GR_FILENAME_PATTERN = re.compile(
    r"^(?P<sample_name>[A-Za-z0-9_]+)-"
    r"(?P<fluence>[-+]?\d*\.?\d+)fs"
    r"hw(?P<delay>[-+]?\d*\.?\d+(?:e[-+]?\d+)?)"
    r"delay(?P<scan_number>\d+)\.gr$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal array utilities (imported by analysis.py and integration.py)
# ---------------------------------------------------------------------------

def _as_image_stack(images, name="images"):
    """Convert a 2D image or 3D image stack into a 3D stack."""
    arr = np.asarray(images, dtype=float)

    if arr.ndim == 2:
        return arr[None, :, :], True
    if arr.ndim == 3:
        return arr, False

    raise ValueError(f"{name} must be 2D or 3D, got shape {arr.shape}")


def _restore_image_dimensionality(image_stack, input_was_2d):
    """Return a 2D image if the original input was 2D, otherwise return the 3D stack."""
    if input_was_2d:
        return image_stack[0]
    return image_stack


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_filename_flexible(file_name, scheme="delay_scan"):
    """
    Parse one filename using a selected naming scheme.

    Parameters
    ----------
    file_name : str or Path
        Filename to parse.
    scheme : str
        Key from FILENAME_PATTERNS.

    Returns
    -------
    dict
        Parsed metadata. Missing fields are not included.
    """
    name = Path(file_name).name

    if scheme not in FILENAME_PATTERNS:
        raise ValueError(
            f"Unknown filename scheme '{scheme}'. "
            f"Available schemes are: {list(FILENAME_PATTERNS)}"
        )

    pattern = FILENAME_PATTERNS[scheme]
    match = pattern.search(name)

    if match is None:
        raise ValueError(f"Could not parse filename with scheme '{scheme}': {name}")

    parsed = match.groupdict()

    for key in ["fluence", "delay", "theta", "degree"]:
        if key in parsed and parsed[key] is not None:
            parsed[key] = float(parsed[key])

    for key in ["image_number", "samz"]:
        if key in parsed and parsed[key] is not None:
            parsed[key] = int(parsed[key])

    parsed["file_name"] = str(file_name)

    return parsed


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


def _parse_gr_filename(file_name):
    """
    Parse one .gr filename and extract sample_name, fluence, delay, and scan number.
    """
    name = Path(file_name).name
    match = _GR_FILENAME_PATTERN.search(name)

    if match is None:
        raise ValueError(f"Could not parse filename: {name}")

    sample_name = match.group("sample_name")
    fluence = float(match.group("fluence"))
    delay = float(match.group("delay"))
    scan_number = int(match.group("scan_number"))

    return sample_name, fluence, delay, scan_number


def _read_gr_file(file_path, comment_chars=("#",)):
    """
    Read a PDFgetX3 .gr file assumed to contain at least two numeric columns:
    r, G(r)

    Returns
    -------
    r : np.ndarray
    gr : np.ndarray
    """
    rows = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if any(line.startswith(c) for c in comment_chars):
                continue

            parts = line.split()
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue

            if len(vals) < 2:
                continue

            rows.append(vals[:2])

    if len(rows) == 0:
        raise ValueError(f"No numeric 2-column data found in file: {file_path}")

    arr = np.array(rows, dtype=float)
    r = arr[:, 0]
    gr = arr[:, 1]

    return r, gr


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def get_image_details(
    folder_path,
    sample_name=None,
    filename_scheme="delay_scan",
    sort=True,
    sort_key="image_number",
    filter_data=False,
    delay_sign=DELAY_SIGN,
    plot=False,
):
    """
    Read TIFF images from a folder and extract filename metadata using a
    flexible filename parser.

    Parameters
    ----------
    folder_path : str or Path
        Folder containing TIFF files.
    sample_name : str or None
        If provided, only keep files whose parsed sample_name matches this value.
        If None, keep all parsed TIFF files.
    filename_scheme : str
        Key from FILENAME_PATTERNS, e.g. "delay_scan" or "theta_samz".
    sort : bool
        If True, sort by sort_key if available.
    sort_key : str
        Metadata field to sort by. Usually "image_number", "delay", or "theta".
    filter_data : bool or list-like
        If False, keep all data. If [min_index, max_index], keep that slice after sorting.
    delay_sign : float
        Multiplier applied to delay if delay exists in parsed metadata.
    plot : bool
        If True, show diagnostic plots.

    Returns
    -------
    dict
        Dictionary containing images, counts, file_names, and all parsed metadata fields.
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")

    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder}")

    file_names = sorted(folder.glob(f"{SCAN_NAME}*.tif"))
    print(f"{len(file_names)} TIFF files found in {folder} with scan name {SCAN_NAME}.")

    if len(file_names) == 0:
        raise ValueError(f"No .tif files found in folder: {folder}")

    metadata = []
    cleaned_files = []

    for file_name in file_names:
        try:
            meta = _parse_filename_flexible(file_name, scheme=filename_scheme)
        except ValueError:
            continue

        if sample_name is not None:
            parsed_sample = meta.get("sample_name", "")
            if parsed_sample.lower() != sample_name.lower():
                continue

        metadata.append(meta)
        cleaned_files.append(str(file_name))

    if len(metadata) == 0:
        raise ValueError(
            f"No TIFF files matched filename_scheme='{filename_scheme}'"
            + (f" and sample_name='{sample_name}'." if sample_name is not None else ".")
        )

    meta_arrays = {}
    all_keys = set()
    for meta in metadata:
        all_keys.update(meta.keys())

    for key in all_keys:
        values = [meta.get(key, np.nan) for meta in metadata]

        if key in ["sample_name", "file_name"]:
            meta_arrays[key] = np.array(values, dtype=str)
        else:
            meta_arrays[key] = np.array(values)

    cleaned_files = np.array(cleaned_files, dtype=str)

    if "delay" in meta_arrays:
        meta_arrays["delay"] = delay_sign * meta_arrays["delay"].astype(float)

    if sort:
        if sort_key in meta_arrays:
            idx_sort = np.argsort(meta_arrays[sort_key])
        elif "image_number" in meta_arrays:
            idx_sort = np.argsort(meta_arrays["image_number"])
        else:
            idx_sort = np.arange(len(cleaned_files))

        for key in meta_arrays:
            meta_arrays[key] = meta_arrays[key][idx_sort]

        cleaned_files = cleaned_files[idx_sort]

    if isinstance(filter_data, (list, tuple, np.ndarray)):
        if len(filter_data) != 2:
            raise ValueError("filter_data must be False or [min_index, max_index].")

        min_val, max_val = filter_data

        if min_val < 0 or max_val > len(cleaned_files):
            raise ValueError("filter_data range is out of bounds.")

        for key in meta_arrays:
            meta_arrays[key] = meta_arrays[key][min_val:max_val]

        cleaned_files = cleaned_files[min_val:max_val]

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
        plt.hist(test.reshape(-1), bins=100, edgecolor="r", histtype="bar", alpha=0.5)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Pixel Number")
        plt.title("Histogram")
        plt.yscale("log")

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=FIGSIZE)
        plt.plot(counts, "o-")
        plt.xlabel("Image index")
        plt.ylabel("Counts")
        plt.title("Total Counts")
        plt.tight_layout()
        plt.show()

    data_dict = {
        "images": data_array,
        "counts": counts,
        "file_names": cleaned_files,
    }

    data_dict.update(meta_arrays)

    return data_dict


def get_images_by_scan_name(
    folder_path,
    scan_name=None,
    sort=True,
    filter_data=False,
    plot=False,
):
    """
    Load TIFF images from a folder.

    If scan_name is provided, only loads files starting with scan_name.
    If scan_name is None, loads all .tif files in the folder.
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")

    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder}")

    if scan_name is None:
        file_names = list(folder.glob("*.tif"))
        print(f"{len(file_names)} TIFF files found in {folder}.")
    else:
        file_names = list(folder.glob(f"{scan_name}*.tif"))
        print(f"{len(file_names)} TIFF files found in {folder} with scan name {scan_name}.")

    if sort:
        file_names = sorted(file_names)

    if len(file_names) == 0:
        if scan_name is None:
            raise ValueError(f"No .tif files found in folder: {folder}")
        else:
            raise ValueError(f"No .tif files found for scan_name='{scan_name}' in {folder}")

    if isinstance(filter_data, (list, tuple, np.ndarray)):
        if len(filter_data) != 2:
            raise ValueError("filter_data must be False or [min_index, max_index].")

        min_val, max_val = filter_data

        if min_val < 0 or max_val > len(file_names):
            raise ValueError("filter_data range is out of bounds.")

        file_names = file_names[min_val:max_val]

    data_array = tf.imread([str(f) for f in file_names])
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
        plt.hist(test.reshape(-1), bins=100, edgecolor="r", histtype="bar", alpha=0.5)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Pixel Number")
        plt.title("Histogram")
        plt.yscale("log")

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=FIGSIZE)
        plt.plot(counts, "o-")
        plt.xlabel("Image index")
        plt.ylabel("Counts")
        plt.title("Total Counts")
        plt.tight_layout()
        plt.show()

    data_dict = {
        "images": data_array,
        "counts": counts,
        "file_names": np.array([str(f) for f in file_names], dtype=str),
    }

    return data_dict


def load_background(
    background_path,
    sort=True,
    plot=False,
    figsize=FIGSIZE,
):
    """
    Load background TIFF image(s) from either a single file or a folder.
    """
    background_path = Path(background_path)

    if not background_path.exists():
        raise ValueError(f"Background path does not exist: {background_path}")

    valid_suffixes = {".tif", ".tiff"}

    if background_path.is_file():
        if background_path.suffix.lower() not in valid_suffixes:
            raise ValueError(f"Background file must be .tif or .tiff, got: {background_path.suffix}")

        files = [background_path]
        background_stack = tf.imread(str(background_path)).astype(float)

        if background_stack.ndim != 2:
            raise ValueError(
                f"Single background file must load as a 2D image, got shape {background_stack.shape}"
            )

        background_stack = background_stack[None, :, :]

    elif background_path.is_dir():
        files = [
            f for f in background_path.iterdir()
            if f.is_file() and f.suffix.lower() in valid_suffixes
        ]

        if sort:
            files = sorted(files)

        if len(files) == 0:
            raise ValueError(f"No .tif or .tiff background files found in folder: {background_path}")

        background_stack = tf.imread([str(f) for f in files]).astype(float)

        if background_stack.ndim == 2:
            background_stack = background_stack[None, :, :]

        if background_stack.ndim != 3:
            raise ValueError(
                f"Loaded background data must be 3D after stacking, got shape {background_stack.shape}"
            )

    else:
        raise ValueError(f"Path is neither a file nor a folder: {background_path}")

    background_mean = np.nanmean(background_stack, axis=0)
    background_std = np.nanstd(background_stack, axis=0)

    if plot:
        first_image = background_stack[0]

        _, axes = plt.subplots(1, 3, figsize=figsize)

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


def save_azimuthal_profiles_to_dat(
    radial,
    profiles,
    file_names,
    output_dir,
    suffix="",
    header="q\tintensity",
    overwrite=False,
):
    """
    Save azimuthally averaged profiles to .dat files using the corresponding
    input file names.

    Parameters
    ----------
    radial : np.ndarray
        1D radial axis of shape (n_q,).
    profiles : np.ndarray
        2D array of shape (n_profiles, n_q).
    file_names : sequence of str or Path
        Original input file names corresponding to each profile.
    output_dir : str or Path
        Directory where .dat files will be written.
    suffix : str, optional
        Suffix appended before '.dat'. Example:
        image001.tif -> image001_azav.dat if suffix="_azav"
    header : str, optional
        Header line written to each file.
    overwrite : bool, optional
        If False, raise an error if an output file already exists.

    Returns
    -------
    saved_files : list of Path
        Paths of written .dat files.
    """
    radial = np.asarray(radial, dtype=float)
    profiles = np.asarray(profiles, dtype=float)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if radial.ndim != 1:
        raise ValueError("radial must be 1D.")
    if profiles.ndim != 2:
        raise ValueError("profiles must be 2D.")
    if profiles.shape[1] != len(radial):
        raise ValueError("profiles.shape[1] must match len(radial).")
    if len(file_names) != profiles.shape[0]:
        raise ValueError("Number of file_names must match number of profiles.")

    saved_files = []

    for i, file_name in enumerate(file_names):
        in_path = Path(file_name)
        out_name = f"{in_path.stem}{suffix}.dat"
        out_path = output_dir / out_name

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {out_path}")

        out_data = np.column_stack((radial, profiles[i]))
        np.savetxt(out_path, out_data, header=header, comments="")
        saved_files.append(out_path)

    return saved_files


def get_gr_details(
    folder_path,
    sample_name=None,
    sort=True,
    filter_data=False,
    delay_sign=1,
    plot=False,
    enforce_same_r=True,
):
    """
    Read .gr files from a folder and extract filename metadata using regex.

    Parameters
    ----------
    folder_path : str or Path
        Folder containing .gr files.
    sample_name : str or None, optional
        If provided, only keep files whose parsed sample_name matches this
        value (case-insensitive).
    sort : bool, optional
        If True, sort data by scan_number.
    filter_data : bool or list-like, optional
        If False, use all data.
        If list-like [min_index, max_index], keep only that slice after sorting.
    delay_sign : int or float, optional
        Multiply parsed delay values by this factor.
    plot : bool, optional
        If True, plot the first G(r) and all delays.
    enforce_same_r : bool, optional
        If True, require all files to have the same r grid.

    Returns
    -------
    dict
        Dictionary containing:
        - "r"           : np.ndarray  (Nr,)
        - "grs"         : np.ndarray  (Nfiles, Nr)
        - "sample_name" : np.ndarray  (Nfiles,)
        - "fluence"     : np.ndarray  (Nfiles,)
        - "delay"       : np.ndarray  (Nfiles,)
        - "scan_number" : np.ndarray  (Nfiles,)
        - "file_names"  : np.ndarray  (Nfiles,)
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")

    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder}")

    file_names = sorted(folder.glob("*.gr"))
    print(f"{len(file_names)} .gr files found in {folder}")

    if len(file_names) == 0:
        raise ValueError(f"No .gr files found in folder: {folder}")

    sample_names = []
    fluence = []
    delay = []
    scan_number = []
    cleaned_files = []

    for file_name in file_names:
        try:
            s_val, f_val, d_val, i_val = _parse_gr_filename(file_name)
        except ValueError:
            continue

        if sample_name is not None and s_val.lower() != sample_name.lower():
            continue

        sample_names.append(s_val)
        fluence.append(f_val)
        delay.append(d_val)
        scan_number.append(i_val)
        cleaned_files.append(str(file_name))

    if len(cleaned_files) == 0:
        if sample_name is None:
            raise ValueError(
                "No .gr files in the folder matched the expected filename pattern."
            )
        else:
            raise ValueError(
                f"No .gr files found for sample_name='{sample_name}' "
                f"that matched the expected filename pattern."
            )

    sample_names = np.array(sample_names, dtype=str)
    fluence = np.array(fluence, dtype=float)
    delay = delay_sign * np.array(delay, dtype=float)
    scan_number = np.array(scan_number, dtype=int)
    cleaned_files = np.array(cleaned_files, dtype=str)

    if sort:
        idx_sort = np.argsort(scan_number)
        sample_names = sample_names[idx_sort]
        fluence = fluence[idx_sort]
        delay = delay[idx_sort]
        scan_number = scan_number[idx_sort]
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
        scan_number = scan_number[min_val:max_val]

    r_ref = None
    gr_list = []

    for file in cleaned_files:
        r, gr = _read_gr_file(file)

        if r_ref is None:
            r_ref = r
        elif enforce_same_r:
            if len(r) != len(r_ref) or not np.allclose(r, r_ref):
                raise ValueError(
                    f"r grid mismatch in file: {file}\n"
                    "Set enforce_same_r=False if you want to handle this manually."
                )

        gr_list.append(gr)

    grs = np.array(gr_list, dtype=float)

    if plot:
        plt.figure(figsize=(7, 5))
        plt.plot(r_ref, grs[0])
        plt.xlabel("r")
        plt.ylabel("G(r)")
        plt.title("First .gr file")
        plt.tight_layout()
        plt.show()

    return {
        "r": r_ref,
        "grs": grs,
        "sample_name": sample_names,
        "fluence": fluence,
        "delay": delay,
        "scan_number": scan_number,
        "file_names": cleaned_files,
    }
