from pathlib import Path
from datetime import date
import time
import h5py
import numpy as np

import trxrd
import globals


def init_h5_file(h5_path, n_images, n_q, n_r):
    """
    Initialize an HDF5 file for batch-processed TRXRD results.

    Parameters
    ----------
    h5_path : str or Path
        Output HDF5 file path.
    n_images : int
        Number of images/profiles to store.
    n_q : int
        Number of q points in azimuthally averaged data.
    n_r : int
        Number of r points in ΔG(r).
    """
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "w") as h5:
        raw = h5.create_group("raw_metadata")
        proc = h5.create_group("processed")
        diff = h5.create_group("difference")

        dt_str = h5py.string_dtype(encoding="utf-8")

        # ------------------------------------------------------------
        # Raw metadata
        # ------------------------------------------------------------
        raw.create_dataset("file_names", shape=(n_images,), dtype=dt_str)
        #raw.create_dataset("sample_name", shape=(n_images,), dtype=dt_str)
        raw.create_dataset("delay", shape=(n_images,), dtype=np.float32)
        raw.create_dataset("image_number", shape=(n_images,), dtype=np.int32)

        # ------------------------------------------------------------
        # Minimal processed outputs needed to build reference later
        # ------------------------------------------------------------
        proc.create_dataset("q", shape=(n_q,), dtype=np.float32)
        proc.create_dataset("center_x", shape=(n_images,), dtype=np.float32)
        proc.create_dataset("center_y", shape=(n_images,), dtype=np.float32)
        proc.create_dataset(
            "profiles_flat",
            shape=(n_images, n_q),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )

        # ------------------------------------------------------------
        # Final difference outputs
        # ------------------------------------------------------------
        diff.create_dataset("reference_profile", shape=(n_q,), dtype=np.float32)
        diff.create_dataset(
            "dI",
            shape=(n_images, n_q),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )
        diff.create_dataset(
            "delta_sq",
            shape=(n_images, n_q),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )
        diff.create_dataset(
            "delta_fq",
            shape=(n_images, n_q),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )
        diff.create_dataset("r", shape=(n_r,), dtype=np.float32)
        diff.create_dataset(
            "delta_gr",
            shape=(n_images, n_r),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )


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
    original file names.

    Parameters
    ----------
    radial : np.ndarray
        1D q axis of shape (n_q,).
    profiles : np.ndarray
        2D array of shape (n_profiles, n_q).
    file_names : sequence of str or Path
        Original input file names corresponding to each profile.
    output_dir : str or Path
        Output directory for .dat files.
    suffix : str, optional
        Suffix appended before '.dat'.
    header : str, optional
        Header line for each .dat file.
    overwrite : bool, optional
        If False, raise if file already exists.

    Returns
    -------
    list of Path
        Written file paths.
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


def process_and_save_chunks(
    data_dict,
    h5_path,
    mask_file,
    group_size=200,
    npt=5000,
    nan_radial_range=(globals.NAN_MIN, globals.NAN_MAX),
    norm_range=(globals.NORM_MIN, globals.NORM_MAX),
    als_lam=globals.LAM_VAL,
    als_p=globals.P_VAL,
    als_niter=10,
    save_azav_dat=False,
    azav_dat_dir=None,
    azav_dat_suffix="",
    azav_dat_overwrite=False,
):
    """
    Process diffraction data in chunks, save minimal pre-reference outputs to HDF5,
    and optionally export azimuthally averaged raw profiles to .dat files.

    Parameters
    ----------
    data_dict : dict
        Dictionary from trxrd.get_image_details / trxrd.remove_counts.
    h5_path : str or Path
        Output HDF5 file path.
    mask_file : str or Path
        Detector mask file.
    group_size : int, optional
        Number of images per processing chunk.
    npt : int, optional
        Number of q bins for azimuthal integration.
    nan_radial_range : tuple, optional
        Range outside which q-space values are set to NaN after integration.
    norm_range : tuple, optional
        q range used for profile normalization.
    als_lam : float, optional
        ALS baseline smoothness parameter.
    als_p : float, optional
        ALS asymmetry parameter.
    als_niter : int, optional
        ALS iteration count.
    save_azav_dat : bool, optional
        If True, save one .dat file per azimuthally averaged raw profile.
    azav_dat_dir : str or Path or None, optional
        Output directory for .dat files.
    azav_dat_suffix : str, optional
        Suffix appended to each .dat filename stem.
    azav_dat_overwrite : bool, optional
        Overwrite existing .dat files if True.
    """
    n_images = len(data_dict["images"])

    groups = list(np.arange(0, n_images, group_size))
    if len(groups) == 0 or groups[-1] != n_images:
        groups.append(n_images)
    groups = np.array(groups)

    q_saved = False

    with h5py.File(h5_path, "a") as h5:
        for i in range(len(groups) - 1):
            start_idx = groups[i]
            stop_idx = groups[i + 1]

            print(f"Started processing files {start_idx} to {stop_idx}")

            # ------------------------------------------------------------
            # Make chunk copy without modifying full data_dict
            # ------------------------------------------------------------
            chunk_dict = {}
            for key, val in data_dict.items():
                if isinstance(val, np.ndarray) and len(val) == n_images:
                    chunk_dict[key] = val[start_idx:stop_idx]
                else:
                    chunk_dict[key] = val

            # ------------------------------------------------------------
            # Mask
            # ------------------------------------------------------------
            print(f"Applying mask for files {start_idx} to {stop_idx}")
            chunk_dict["images"] = trxrd.apply_nan_mask(
                chunk_dict["images"],
                mask_path=mask_file,
                plot=False,
                image_index=0,
            )

            # ------------------------------------------------------------
            # Remove hot pixels
            # ------------------------------------------------------------
            print(f"Removing hot pixels for files {start_idx} to {stop_idx}")
            xr_result = trxrd.remove_xrays_pool(
                chunk_dict["images"],
                std_factor=globals.STD_FACTOR,
                plot=False,
                image_index=0,
                return_dict=True,
                max_workers=globals.MAX_PROCESSORS,
                progress_interval=100,
            )
            chunk_dict["images"] = xr_result["clean_data"]

            # ------------------------------------------------------------
            # Find centers
            # ------------------------------------------------------------
            print(f"Finding centers for files {start_idx} to {stop_idx}")
            center_result = trxrd.find_centers_in_stack_radial_parallel(
                data_array=chunk_dict["images"],
                center_guess=(globals.CENTER_Y, globals.CENTER_X),
                search_radius=10,
                center_mask=None,
                r_min=0,
                r_max=1600,
                downsample=1,
                intensity_threshold=None,
                top_percentile=60,
                max_workers=globals.MAX_PROCESSORS,
                progress_interval=100,
                plot_example=False,
                example_index=0,
                plot_center_vs_image=False,
            )

            centers = center_result["centers_xy"]

            # ------------------------------------------------------------
            # Azimuthal integration
            # ------------------------------------------------------------
            print(f"Calculating azimuthal averages for files {start_idx} to {stop_idx}")
            az_result = trxrd.azimuthal_average_pyfai(
                images=chunk_dict["images"],
                centers=centers,
                npt=npt,
                unit="q_A^-1",
                nan_radial_range=nan_radial_range,
                azimuth_range=None,
                integration_mask=None,
                return_dict=True,
                progress_interval=100,
            )

            q = az_result["radial"]
            profiles = az_result["profiles"]

            # ------------------------------------------------------------
            # Optional .dat save of raw integrated profiles
            # ------------------------------------------------------------
            if save_azav_dat:
                if azav_dat_dir is None:
                    raise ValueError(
                        "azav_dat_dir must be provided when save_azav_dat=True."
                    )

                save_azimuthal_profiles_to_dat(
                    radial=q,
                    profiles=profiles,
                    file_names=chunk_dict["file_names"],
                    output_dir=azav_dat_dir,
                    suffix=azav_dat_suffix,
                    header="q\tintensity",
                    overwrite=azav_dat_overwrite,
                )

            # ------------------------------------------------------------
            # Normalize profiles
            # ------------------------------------------------------------
            print(f"Normalizing profiles for files {start_idx} to {stop_idx}")
            norm_result = trxrd.normalize_profiles_to_range(
                radial=q,
                profiles=profiles,
                norm_range=norm_range,
                mode="mean",
                plot=False,
                return_dict=True,
            )
            profiles_norm = norm_result["normalized_profiles"]

            # ------------------------------------------------------------
            # Subtract ALS baseline
            # ------------------------------------------------------------
            print(f"Subtracting baseline for files {start_idx} to {stop_idx}")
            als_result = trxrd.subtract_als_baseline(
                data_array=profiles_norm,
                x_vals=q,
                lam=als_lam,
                p=als_p,
                niter=als_niter,
                plot=False,
                return_dict=True,
            )
            profiles_flat = als_result["corrected_data"]

            # ------------------------------------------------------------
            # Save metadata
            # ------------------------------------------------------------
            h5["raw_metadata/file_names"][start_idx:stop_idx] = chunk_dict["file_names"].astype(str)
            #h5["raw_metadata/sample_name"][start_idx:stop_idx] = chunk_dict["sample_name"].astype(str)
            h5["raw_metadata/delay"][start_idx:stop_idx] = chunk_dict["delay"].astype(np.float32)
            h5["raw_metadata/image_number"][start_idx:stop_idx] = chunk_dict["image_number"].astype(np.int32)

            # ------------------------------------------------------------
            # Save processed outputs
            # ------------------------------------------------------------
            if not q_saved:
                h5["processed/q"][:] = q.astype(np.float32)
                q_saved = True
            else:
                q_existing = h5["processed/q"][:]
                if not np.allclose(q_existing, q, rtol=1e-6, atol=1e-8, equal_nan=True):
                    raise ValueError("q axis changed between chunks. This should not happen.")

            h5["processed/center_x"][start_idx:stop_idx] = centers[:, 0].astype(np.float32)
            h5["processed/center_y"][start_idx:stop_idx] = centers[:, 1].astype(np.float32)
            h5["processed/profiles_flat"][start_idx:stop_idx, :] = profiles_flat.astype(np.float32)


def compute_reference_and_differences(
    h5_path,
    composition="BaTiO3",
    reference_selector=None,
    r_max=20.0,
    n_r=2000,
    q_range_for_pdf=(0.5, 16.0),
):
    """
    Build a reference profile from processed profiles, compute dI, ΔS(Q), ΔF(Q),
    and ΔG(r), then append them to the HDF5 file.

    Parameters
    ----------
    h5_path : str or Path
        Input/output HDF5 file.
    composition : str or dict, optional
        Chemical composition used for form-factor normalization.
    reference_selector : array-like, callable, or None, optional
        Reference selector passed to trxrd.make_reference_profile.
        If None, uses all delays < 0.
    r_max : float, optional
        Maximum r value for ΔG(r).
    n_r : int, optional
        Number of r points.
    q_range_for_pdf : tuple, optional
        Q range used for ΔF(Q) → ΔG(r) transform.
    """
    with h5py.File(h5_path, "a") as h5:
        q = h5["processed/q"][:]
        profiles_flat = h5["processed/profiles_flat"][:]
        delays = h5["raw_metadata/delay"][:]

        print("Building reference profile")
        ref_result = trxrd.make_reference_profile(
            profiles=profiles_flat,
            delays=delays,
            reference_selector=reference_selector,
            return_dict=True,
        )
        reference_profile = ref_result["reference_profile"].astype(np.float32)
        h5["difference/reference_profile"][:] = reference_profile

        print("Computing dI")
        di_result = trxrd.compute_delta_profiles(
            profiles=profiles_flat,
            reference_profile=reference_profile,
            mode="subtract",
            return_dict=True,
        )
        dI = di_result["delta_profiles"].astype(np.float32)
        h5["difference/dI"][:, :] = dI

        print("Normalizing to ΔS(Q) and ΔF(Q)")
        fq_result = trxrd.normalize_xray_scattering_to_sq_fq(
            q=q,
            iq=dI,
            composition=composition,
            mode="difference",
            plot=False,
            return_dict=True,
        )
        delta_sq = fq_result["delta_sq"].astype(np.float32)
        delta_fq = fq_result["delta_fq"].astype(np.float32)

        h5["difference/delta_sq"][:, :] = delta_sq
        h5["difference/delta_fq"][:, :] = delta_fq

        print("Computing ΔG(r)")
        gr_result = trxrd.compute_delta_gr_from_delta_fq(
            q=q,
            delta_fq=delta_fq,
            r_max=r_max,
            n_r=n_r,
            q_range=q_range_for_pdf,
            window="lorch",
            plot=False,
            return_dict=True,
        )

        h5["difference/r"][:] = gr_result["r"].astype(np.float32)
        h5["difference/delta_gr"][:, :] = gr_result["delta_gr"].astype(np.float32)


if __name__ == "__main__":
    # ------------------------------------------------------------
    # User inputs
    # ------------------------------------------------------------
    data_path = globals.DATA_PATH
    scan_name = globals.SCAN_NAME
    mask_file = globals.MASK_FILE

    exp_label = globals.SCAN_NAME
    save_path = globals.SAVE_PATH
    h5_path = save_path / f"{exp_label}.h5"

    # ------------------------------------------------------------
    # Optional .dat export of azimuthally averaged raw profiles
    # ------------------------------------------------------------
    save_azav_dat = globals.SAVE_AZAV_DAT
    azav_dat_dir = save_path / f"{exp_label}_azav_dat"
    azav_dat_suffix = ""
    azav_dat_overwrite = True

    # ------------------------------------------------------------
    # Slice / processing settings
    # ------------------------------------------------------------
    min_idx = 0
    max_idx = 2000
    group_size = 200

    npt = 5000
    n_r = 2000

    start = time.perf_counter()

    # ------------------------------------------------------------
    # Load and filter metadata/images
    # ------------------------------------------------------------
    print("Loading diffraction signal")
    data_dict = trxrd.get_image_details(
        folder_path=data_path,
        sample_name=scan_name,
        sort=True,
        filter_data=False,
        plot=False,
    )

    print("Filtering based on counts")
    count_result = trxrd.remove_counts(
        data_dict,
        std_factor=globals.STD_FACTOR,
        added_range=None,
        plot=False,
        return_dict=True,
    )
    data_dict = count_result["filtered_data"]
    data_dict.pop("sample_name", None)

    if max_idx > len(data_dict["images"]):
        max_idx = len(data_dict["images"])

    current_len = len(data_dict["images"])
    for key, val in data_dict.items():
        if isinstance(val, np.ndarray) and len(val) == current_len:
            data_dict[key] = val[min_idx:max_idx]

    n_images = len(data_dict["images"])
    print(f"Working on {n_images} filtered images")

    # ------------------------------------------------------------
    # Initialize HDF5
    # ------------------------------------------------------------
    init_h5_file(
        h5_path=h5_path,
        n_images=n_images,
        n_q=npt,
        n_r=n_r,
    )

    # ------------------------------------------------------------
    # First pass: preprocess and save minimal per-image data
    # ------------------------------------------------------------
    process_and_save_chunks(
        data_dict=data_dict,
        h5_path=h5_path,
        mask_file=mask_file,
        group_size=group_size,
        npt=npt,
        nan_radial_range=(globals.NAN_MIN, globals.NAN_MAX),
        norm_range=(globals.NORM_MIN, globals.NORM_MAX),
        als_lam=globals.LAM_VAL,
        als_p=globals.P_VAL,
        als_niter=10,
        save_azav_dat=save_azav_dat,
        azav_dat_dir=azav_dat_dir,
        azav_dat_suffix=azav_dat_suffix,
        azav_dat_overwrite=azav_dat_overwrite,
    )

    # ------------------------------------------------------------
    # Second pass: build reference and append difference outputs
    # ------------------------------------------------------------
    compute_reference_and_differences(
        h5_path=h5_path,
        composition=globals.COMPOSITION,
        reference_selector=None,   # default uses delays < 0
        r_max=globals.R_MAX,
        n_r=globals.N_R,
        q_range_for_pdf=(globals.Q_MIN, globals.Q_MAX),
    )

    elapsed = time.perf_counter() - start
    print(f"Finished. Total elapsed time: {elapsed:.1f} s")
    print(f"Wrote results to: {h5_path}")