# trxrd/__init__.py
# Re-exports the full public API so notebooks can use `import trxrd` unchanged.

from trxrd.io import (
    FILENAME_PATTERNS,
    get_image_details,
    get_images_by_scan_name,
    load_background,
    save_azimuthal_profiles_to_dat,
    get_gr_details,
)

from trxrd.masking import (
    make_circular_mask,
    load_detector_mask,
    build_combined_mask,
    apply_mask_from_bool,
    apply_beamstop_mask,
    build_pyfai_mask,
    apply_nan_mask,
)

from trxrd.integration import (
    yx_to_xy,
    xy_to_yx,
    normalize_centers,
    tilt_to_rotations,
    make_azimuthal_integrator,
    custom_polarization_map_notebook,
    find_diffraction_center_from_guess_radial_fast,
    find_centers_in_stack_radial_parallel,
    azimuthal_average_pyfai,
    get_polar_map,
    azimuthal_anisotropy,
)

from trxrd.normalization import (
    compute_background_azimuthal_average,
    subtract_scaled_background_profile,
    plot_normalization_window,
    normalize_profiles_to_range,
    subtract_als_baseline,
    apply_polynomial_baseline,
)

from trxrd.pdf import (
    load_form_factor_table,
    load_form_factor,
    parse_composition_string,
    compute_average_form_factors,
    fit_iq_to_f2_high_q,
    correct_iq,
    normalize_xray_scattering_to_sq_fq,
    compute_delta_gr_from_delta_fq,
    compute_qualitative_difference_pdf,
)

from trxrd.analysis import (
    remove_counts,
    remove_xrays,
    remove_xrays_pool,
    average_images_by_delay,
    average_profiles_by_delay,
    make_reference_profile,
    compute_delta_profiles,
    lineouts_by_delay_from_per_image_profiles,
    apply_gaussian_smoothing,
    make_reference_gr,
    compute_delta_grs,
    average_delta_grs_by_delay,
)
