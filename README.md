# TRXRD

**TRXRD** is a Python package for processing and analyzing time-resolved X-ray scattering (TR-XRD) and total scattering datasets, especially those collected at the 7ID station at the Advanced Photon Source. It provides tools for batch image processing, masking, normalization, background subtraction, azimuthal integration, and pair distribution function (PDF) analysis, enabling efficient extraction of time-dependent structural information.

---

## Features

* Batch processing of 2D detector images (TIFF format)
* Beam stop and detector masking
* X-ray removal and outlier filtering
* Automatic and parallel diffraction center detection
* Azimuthal integration via pyFAI with polarization correction
* Background subtraction and normalization
* Time-resolved data grouping and averaging by delay
* Temperature-dependent scan support
* Singular value decomposition (SVD) of difference data
* Peak tracking and q-axis drift correction
* Full PDF workflow: I(Q) → S(Q) → F(Q) → G(r) and ΔG(r)
* PDF fitting via diffpy-CMI
* Export to `.dat`, `.gr`, and `.h5` formats

---

## Installation

### Clone the repository

```bash
git clone https://github.com/lheald2/trxrd.git
cd trxrd
```

### Install dependencies

```bash
conda env create -f environment.yml
conda activate trxrd
```

Or install manually:

```bash
pip install numpy scipy matplotlib h5py pyFAI tifffile diffpy.structure diffpy.srfit
```

---

## Quick Start

All notebooks and batch scripts pull configuration from `globals.py`. Set your experimental parameters there before running any processing.

**Recommended workflow for a new dataset:**

1. Use `Find_Centers.ipynb` to find and save accurate diffraction centers.
2. Use `Data_Analysis.ipynb` to interactively tune all processing parameters and update `globals.py`.
3. Run `process_data.py` (or `batch_process_dat_files.py`) for full batch processing.
4. Visualize results in `Analyze_Processed_Data.ipynb`.

For a faster first look, `Fast_Analysis.ipynb` averages by delay before full processing to reduce compute time.

---

## Workflow Overview

```
TIFF Images (2D diffraction)
    ↓  remove hot pixels, filter by count statistics
Clean Images
    ↓  find diffraction center, azimuthal integration, polarization correction
1D Profiles  I(Q)
    ↓  subtract background, normalize
Normalized Profiles
    ↓  average by time delay, compute difference vs. reference
ΔI(Q) / ΔI(Q)/I(Q)
    ↓  SVD, peak tracking, lineouts, scale drift correction
Analysis Results

Optional PDF Path:
1D Profiles  I(Q)
    ↓  form factor correction, normalize
S(Q)  →  F(Q)
    ↓  Fourier transform
G(r)  /  ΔG(r)
    ↓  diffpy-CMI fitting
Structural Insights
```

---

## Package Modules (`trxrd/`)

### `trxrd/io.py`

File I/O and metadata parsing for all supported data formats.

* `get_image_details()` — load TIFF image stacks with flexible filename parsing and metadata extraction
* `get_images_by_scan_name()` — load images filtered by scan name prefix
* `load_background()` — load background image(s) and compute mean/std
* `read_tif_metadata()` — parse QXRD `.metadata` sidecar files
* `save_azimuthal_profiles_to_dat()` — export 1D profiles to `.dat` files
* `get_gr_details()` / `get_grs_by_scan_name()` — load pair distribution function `.gr` files
* `get_dat_details()` — load `.dat` files (Q, I(Q) pairs)

Supports three scan types via `FILENAME_PATTERNS`: `delay_scan`, `theta_samz`, and `temp_dep`.

### `trxrd/masking.py`

Create and apply masks to detector images.

* `make_circular_mask()` — create a circular beam stop mask
* `load_detector_mask()` — load a pre-computed detector mask from file
* `build_combined_mask()` — merge beam stop and detector masks with visualization
* `apply_mask_from_bool()` — apply a boolean mask to 2D or 3D image data (fills with NaN)
* `build_pyfai_mask()` — create a pyFAI-compatible mask array

### `trxrd/integration.py`

Azimuthal averaging and diffraction center finding using pyFAI.

* `make_azimuthal_integrator()` — create a pyFAI `AzimuthalIntegrator` from a PONI file or manual parameters
* `azimuthal_average_pyfai()` — perform 1D azimuthal averaging with masking and polarization correction
* `find_diffraction_center_from_guess_radial_fast()` — optimize center position by radial profile sharpness
* `find_centers_in_stack_radial_parallel()` — parallel center finding across an image stack
* `custom_polarization_map_notebook()` — apply a custom polarization correction factor
* `get_polar_map()` — extract 2D polar/detector space maps
* `azimuthal_anisotropy()` — compute directional scattering anisotropy

### `trxrd/analysis.py`

Core data processing and statistical analysis.

* **Image cleaning:** `remove_xrays()`, `remove_xrays_pool()`, `remove_counts()` — hot pixel removal and count-based outlier filtering
* **Grouping & averaging:** `average_images_by_delay()`, `average_profiles_by_delay()`, `average_grs_by_temperature()`, `average_iqs_by_temperature()`
* **Difference data:** `make_reference_profile()`, `make_reference_gr()`, `compute_delta_profiles()`, `compute_delta_grs()`, `average_delta_grs_by_delay()`
* **SVD:** `svd_analysis()` — singular value decomposition of ΔI(Q) matrices
* **Peak analysis:** `detect_peaks()`, `fit_one_peak()`, `track_peaks()`, `plot_drift()`, `plot_diagnostics()`
* **Drift correction:** `scale_drift_factor()`, `apply_scale_correction()`, `verify_correction()` — correct q-axis drift across a time series
* **Utilities:** `apply_gaussian_smoothing()`, `lineouts_by_delay_from_per_image_profiles()`

### `trxrd/normalization.py`

Background subtraction and baseline correction.

* `compute_background_azimuthal_average()` — azimuthally average a background image with automatic center finding
* `subtract_scaled_background_profile()` — subtract a scaled background from profiles
* `normalize_profiles_to_range()` — clip and normalize profiles to a specified Q range
* `subtract_als_baseline()` — asymmetric least squares (ALS) baseline removal
* `apply_polynomial_baseline()` — fit and subtract a polynomial baseline
* `plot_normalization_window()` — visualize normalization windows

### `trxrd/pdf.py`

X-ray form factors and pair distribution function calculations.

* `load_form_factor_table()` / `load_form_factor()` — load atomic x-ray form factor coefficients from CSV
* `parse_composition_string()` — parse chemical formulas (e.g., `"BaTiO3"`)
* `compute_average_form_factors()` — compute composition-weighted average form factors
* `correct_iq()` / `normalize_xray_scattering_to_sq_fq()` — convert I(Q) to S(Q) and F(Q) with form factor corrections
* `compute_delta_gr_from_delta_fq()` — Fourier transform ΔF(Q) → ΔG(r)
* `compute_qualitative_difference_pdf()` — compute unnormalized difference PDF

---

## Configuration: `globals.py`

Central configuration file for all experimental parameters. Edit this file before running any processing scripts or notebooks.

| Category | Variables |
|---|---|
| Paths | `DATA_PATH`, `MASK_FILE`, `BACKGROUND_PATH`, `SAVE_PATH` |
| Scan | `SCAN_NAME`, `SCAN_TYPE`, `DELAY_SIGN` |
| Detector geometry | `PIXEL1`, `PIXEL2`, `DISTANCE`, `WAVELENGTH`, `PONI_FILE`, `TILT_ANGLE`, `TILT_PLANE_ROTATION`, `ROT3` |
| Masking | `MASK_CENTER_X`, `MASK_CENTER_Y`, `MASK_RADIUS` |
| Processing | `CENTER_X`, `CENTER_Y`, `DOWNSAMPLE`, `NAN_MIN`, `NAN_MAX`, `NORM_MIN`, `NORM_MAX` |
| PDF | `COMPOSITION`, `R_MAX`, `N_R`, `Q_MIN`, `Q_MAX`, `WINDOW`, `FORM_FACTOR_FILE` |

---

## Batch Processing Scripts

### `process_data.py`

Full processing pipeline (equivalent to `Data_Analysis.ipynb`) without visualization. Reads TIFF images, applies all processing steps, and saves results to an HDF5 file. Run this after finalizing parameters in `globals.py`.

```bash
python process_data.py
```

### `batch_process_dat_files.py`

Reads TIFF images, applies masks and cleanup, and saves normalized azimuthal averages as individual `.dat` files for use with external programs like *PDFgetX3*.

```bash
python batch_process_dat_files.py
```

### `batch_process_to_gr.py`

Converts azimuthal profiles (`.dat`) to pair distribution functions (`.gr`) by applying form factor corrections and a Fourier transform.

```bash
python batch_process_to_gr.py
```

### `process_Temp_Dep.py`

Batch processor for temperature-dependent (non-time-resolved) scans. Groups and averages data by temperature rather than delay.

```bash
python process_Temp_Dep.py
```

### `diffpy_fitting.py`

Fits PDFs using diffpy-CMI. Requires a structural model (CIF file) and processed `.gr` files.

---

## Jupyter Notebooks

| Notebook | Purpose |
|---|---|
| `Data_Analysis.ipynb` | Interactive step-by-step processing with visualization; use to tune `globals.py` |
| `Fast_Analysis.ipynb` | Quick analysis — averages by delay first, then processes; good for a fast first look |
| `Find_Centers.ipynb` | Find precise diffraction centers; run once per dataset and save to `globals.py` |
| `Analyze_Processed_Data.ipynb` | Visualize and analyze HDF5 output from `process_data.py` |
| `Azimuthal_Integration_Testing.ipynb` | Test and validate azimuthal integration parameters and detector geometry |
| `Diagnostics.ipynb` | Diagnostic plots for assessing data quality (hot pixels, count statistics, masks) |
| `Process_Grs.ipynb` | Process and visualize `.gr` PDF files from PDFgetX3 |
| `Fitting_Notebook.ipynb` | PDF fitting workflows using diffpy-CMI |
| `Process_Temp_Dep.ipynb` | Interactive processing for temperature-dependent scans |
| `Non_Scan_Data.ipynb` | Handle single-image or non-scan diffraction data |

---

## Performance Tips

* Use precomputed combined masks to avoid recomputation on every run
* Use `find_centers_in_stack_radial_parallel()` to parallelize center finding
* Use HDF5 for storing large processed datasets
* Run `Fast_Analysis.ipynb` for an initial look before committing to full per-image processing

---

## Dependencies

* numpy
* scipy
* matplotlib
* h5py
* pyFAI
* tifffile
* diffpy.structure
* diffpy.srfit (for PDF fitting)

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a new branch
3. Submit a pull request

---

## License

MIT License

---

## Contact

**Lauren Heald**
Email: [lauren.f.heald@gmail.com](mailto:lauren.f.heald@gmail.com)

---

## Acknowledgments

Dr. Burak Guzelturk — APS Physicist
