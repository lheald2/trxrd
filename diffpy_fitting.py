"""
PDF fitting module for time-resolved XRD experiments.

Public API
----------
FitConfig            -- dataclass holding all fitting configuration
FitResult            -- dataclass returned by fit_one; has .plot() and .to_dict()
fit_one              -- fit a single .gr file, returns FitResult
fit_batch            -- fit a list/glob of .gr files with full/reduced mode splitting
results_to_dataframe -- convert a list of FitResults to a DataFrame
plot_series          -- plot all parameters vs file index from a DataFrame or CSV path
normalize_full_indices   -- convert full_indices spec to a set of ints
average_reference_params -- average numeric params across full-mode result dicts
save_reference_params    -- write a parameter dict to JSON

ΔG(r) fitting (new):
Reference            -- bundle of averaged G(r) + reference fit params
build_reference      -- average pre-t0 .gr files and full-fit them
fit_dgr              -- single ΔG(r) fit
fit_dgr_batch        -- batch ΔG(r) fits against a Reference

Example (notebook)
------------------
from diffpy_fitting import FitConfig, fit_one, fit_batch, results_to_dataframe, plot_series

cfg = FitConfig(
    cif_path="BaTiO3_tetragonal.cif",
    rmin=10, rmax=30, qmax=12,
    save_outputs=False,         # skip writing files during exploration
)

# single file
result = fit_one("scan_001.gr", cfg)
print(result.rw, result.params)
result.plot()

# batch
results = fit_batch("data/*.gr", cfg, full_indices=(1, 10))
df = results_to_dataframe(results)
plot_series(df)

# ΔG(r) workflow
from diffpy_fitting import build_reference, fit_dgr_batch

ref = build_reference(["pre_t0_001.gr", "pre_t0_002.gr", "pre_t0_003.gr"], cfg)
dgr_results = fit_dgr_batch("data/*.gr", ref, cfg)
dgr_df = results_to_dataframe(dgr_results)
plot_series(dgr_df)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from diffpy.srfit.fitbase import FitContribution, FitRecipe, FitResults, Profile
from diffpy.srfit.pdf import PDFParser, PDFGenerator
from diffpy.structure.parsers import get_parser
from diffpy.srfit.structure import constrainAsSpaceGroup


# ============================================================
# Default parameter bounds (used when building FitConfig)
# ============================================================

_DEFAULT_PARAM_BOUNDS: Dict[str, Tuple] = {
    "s1":           (1e-4, 10.0),
    "Delta2":       (0.0,  5.0),
    "Calib_Qdamp":  (0.0,  0.15),
    "Calib_Qbroad": (0.0,  0.10),
    "a":            (3.95, 4.05),
    "c":            (4.00, 4.10),
    "_adp_default": (1e-5, 0.05),  # applied to any ADP_ parameter
}

_DEFAULT_PINNED = ["Delta2", "Calib_Qdamp", "Calib_Qbroad"]


# ============================================================
# Configuration
# ============================================================

@dataclass
class FitConfig:
    """
    All configuration for a PDF fitting run.

    Parameters
    ----------
    cif_path : str or Path
        Path to the structure CIF file.
    rmin, rmax, rstep : float
        r-range for the PDF fit (Angstroms).
    qmax, qmin : float
        Q-range used in the PDF calculation (Angstroms^-1).
    space_group : str
        Space group symbol passed to constrainAsSpaceGroup.
    scale_i, uiso_i, delta2_i, qdamp_i, qbroad_i : float
        Initial parameter guesses.
    fix_qdamp, fix_qbroad : bool
        Pin instrumental parameters even in full-mode fits (e.g. if
        pre-calibrated). In reduced mode they are always pinned regardless.
    param_bounds : dict
        Bounds for least_squares (method='trf'). Keys are parameter names;
        '_adp_default' applies to any parameter whose name starts with 'ADP_'.
        Format: {name: (lower, upper)}, use None for one-sided unbounded.
    pinned_in_reduced : list of str
        Parameters held fixed in reduced-mode fits (seeded from reference avg).
    save_outputs : bool
        If False, skip writing .fit / .res / figure files entirely.
        Useful for interactive exploration in a notebook.
    resdir, fitdir, figdir : Path or None
        Output directories. When None, defaults to <gr_file_parent>/res|fit|fig.
        Ignored when save_outputs=False.
    run_parallel : bool
        Use multiprocessing for the PDF calculation.
    """
    cif_path: Union[str, Path]

    rmin:  float = 10.0
    rmax:  float = 30.0
    rstep: float = 0.01
    qmax:  float = 12.0
    qmin:  float = 0.3

    space_group: str = "P4mm"

    scale_i:  float = 0.5
    uiso_i:   float = 0.005
    delta2_i: float = 1.7
    qdamp_i:  float = 0.04
    qbroad_i: float = 0.02

    fix_qdamp:  bool = False
    fix_qbroad: bool = False

    param_bounds:      Dict         = field(default_factory=lambda: dict(_DEFAULT_PARAM_BOUNDS))
    pinned_in_reduced: List[str]    = field(default_factory=lambda: list(_DEFAULT_PINNED))

    save_outputs: bool          = True
    resdir:       Optional[Path] = None
    fitdir:       Optional[Path] = None
    figdir:       Optional[Path] = None

    run_parallel: bool = True

    def __post_init__(self):
        self.cif_path = Path(self.cif_path)


# ============================================================
# Result object
# ============================================================

@dataclass
class FitResult:
    """
    Result from a single PDF fit.

    Attributes
    ----------
    params : dict
        All parameter values (free and fixed) after fitting.
    rw : float
        Weighted residual — primary goodness-of-fit metric.
    mode : str
        'full', 'reduced', or 'dgr'.
    gr_path : Path
        The .gr file that was fit.
    recipe : FitRecipe
        The fitted recipe object, kept for advanced inspection.
    """
    params:   Dict
    rw:       float
    mode:     str
    gr_path:  Path
    recipe:   FitRecipe

    def plot(self, show: bool = True) -> "plt.Figure":
        """Plot the fitted PDF with residual. Returns the Figure."""
        if self.mode == "dgr":
            return _plot_dgr_fit(self.recipe, show=show)
        return _plot_fit(self.recipe, show=show)

    def to_dict(self) -> Dict:
        """Flat dict with file, rw, mode, and all parameters — one CSV row."""
        return {"file": self.gr_path.name, "rw": self.rw, "mode": self.mode,
                **self.params}


# ============================================================
# Reference object (for ΔG fitting)
# ============================================================

@dataclass
class Reference:
    """
    Bundle of everything needed as a ΔG-fit baseline.

    Attributes
    ----------
    r : np.ndarray
        r-grid of the averaged reference G(r).
    g : np.ndarray
        Averaged G(r) array (data-based reference).
    params : dict
        Parameters from a full-mode fit of the averaged G(r).
    rw : float
        Rw of the reference full fit.
    source_paths : list of Path
        The .gr files that were averaged.
    recipe : FitRecipe or None
        The recipe used for the reference full fit (kept for inspection).
    """
    r:            np.ndarray
    g:            np.ndarray
    params:       Dict
    rw:           float
    source_paths: List[Path]
    recipe:       Optional[FitRecipe] = None


# ============================================================
# Public fitting API
# ============================================================

def fit_one(
    gr_path: Union[str, Path],
    config: FitConfig,
    mode: str = "full",
    initial_params: Optional[Dict] = None,
    reference_params: Optional[Dict] = None,
    verbose: bool = True,
) -> FitResult:
    """
    Fit a single .gr file and return a FitResult.

    Parameters
    ----------
    gr_path : str or Path
    config : FitConfig
    mode : {'full', 'reduced'}
        'full'    — staged refinement: scale → lat → ADPs → Delta2 → inst.
        'reduced' — refines only scale, lattice, and ADPs; pins Delta2 and
                    instrumental params to reference_params values.
    initial_params : dict, optional
        Warm-start values. Falls back to config defaults for missing keys.
    reference_params : dict, optional
        Required when mode='reduced'. Provides the pinned param values.
    verbose : bool
        If True, print Rw after each refinement stage.

    Returns
    -------
    FitResult
    """
    gr_path = Path(gr_path)
    if initial_params is None:
        initial_params = {}

    if verbose:
        print(f"Fitting {gr_path.name} [{mode}]")

    if mode == "reduced":
        if reference_params is None:
            raise ValueError("reduced mode requires reference_params")
        seed = dict(initial_params)
        for pname in config.pinned_in_reduced:
            if pname in reference_params:
                seed[pname] = reference_params[pname]
        recipe = _make_recipe(config, gr_path, seed, fix_instrumental=True)
        recipe.fithooks[0].verbose = 0
        _fit_reduced(recipe, config, verbose=verbose)
    elif mode == "full":
        recipe = _make_recipe(config, gr_path, initial_params)
        recipe.fithooks[0].verbose = 0
        _fit_full(recipe, config, verbose=verbose)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    res = FitResults(recipe)

    if config.save_outputs:
        _save_fit_outputs(recipe, res, gr_path, config)

    params = {name: recipe.get(name).value
              for name in recipe._parameters.keys()}

    return FitResult(params=params, rw=res.rw, mode=mode,
                     gr_path=gr_path, recipe=recipe)


def fit_batch(
    gr_paths: Union[str, Path, List[Union[str, Path]]],
    config: FitConfig,
    full_indices: Union[Tuple, List] = (1, 1),
    save_reference: bool = True,
) -> List[FitResult]:
    """
    Fit a collection of .gr files using the two-pass full/reduced strategy.

    Pass 1 fits the files in full_indices with staged refinement and averages
    their results into a reference. Pass 2 fits all remaining files in reduced
    mode, pinning Delta2 and instrumental params to the reference average.

    Parameters
    ----------
    gr_paths : str, Path, or list
        Glob pattern string (e.g. "data/*.gr"), a directory (fits all *.gr
        inside), or an explicit list of paths.
    config : FitConfig
    full_indices : tuple or list
        Which sorted file indices (1-based) get full-mode refinement.
        Accepts (start, end), [(start, end), ...], [i, j, ...], or mixed.
    save_reference : bool
        Write the averaged reference parameters to JSON if save_outputs=True.

    Returns
    -------
    list of FitResult, sorted by original file index.
    """
    files = _resolve_gr_paths(gr_paths)
    if not files:
        raise FileNotFoundError(f"No .gr files found for: {gr_paths!r}")

    full_set = normalize_full_indices(full_indices, len(files))
    if not full_set:
        raise ValueError("full_indices produced no full-mode files; at least one is required.")

    full_files    = [(i, p) for i, p in enumerate(files, 1) if i in full_set]
    reduced_files = [(i, p) for i, p in enumerate(files, 1) if i not in full_set]

    print(f"Found {len(files)} file(s): {len(full_files)} full, {len(reduced_files)} reduced.")

    # ---- Pass 1: full fits ----
    print("\n=== Pass 1: Full-mode refinements ===")
    full_results: List[FitResult] = []
    params: Dict = {}
    for idx, gr_path in full_files:
        print(f"  [{idx}/{len(files)}] {gr_path.name}...", end=" ", flush=True)
        try:
            result = fit_one(gr_path, config, mode="full", initial_params=params, verbose=False)
            params = result.params
            full_results.append(result)
            print(f"Rw = {result.rw:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")

    if not full_results:
        raise RuntimeError("No full-mode fits succeeded; cannot build reference.")

    # ---- Build reference average ----
    reference_params = average_reference_params([r.to_dict() for r in full_results])

    if save_reference and config.save_outputs:
        ref_dir = Path(config.resdir) if config.resdir else full_files[0][1].parent / "res"
        ref_dir.mkdir(exist_ok=True)
        save_reference_params(reference_params, ref_dir / "reference_params.json")

    print("\nReference values (averaged over full fits):")
    for k in config.pinned_in_reduced:
        if k in reference_params:
            print(f"  {k:14s} = {reference_params[k]:.6f}")

    # ---- Pass 2: reduced fits ----
    reduced_results: List[FitResult] = []
    if reduced_files:
        print("\n=== Pass 2: Reduced-mode refinements ===")
        params = dict(reference_params)
        for idx, gr_path in reduced_files:
            print(f"  [{idx}/{len(files)}] {gr_path.name}...", end=" ", flush=True)
            try:
                result = fit_one(gr_path, config, mode="reduced",
                                 initial_params=params,
                                 reference_params=reference_params,
                                 verbose=False)
                params = result.params
                reduced_results.append(result)
                print(f"Rw = {result.rw:.4f}")
            except Exception as e:
                print(f"FAILED: {e}")

    # Merge and restore original file order
    indexed = (
        [(i, r) for (i, _), r in zip(full_files,    full_results)]    +
        [(i, r) for (i, _), r in zip(reduced_files, reduced_results)]
    )
    indexed.sort(key=lambda x: x[0])
    return [r for _, r in indexed]


def results_to_dataframe(results: List[FitResult]) -> pd.DataFrame:
    """
    Convert a list of FitResults to a DataFrame with one row per file.

    Columns: 'file', 'rw', 'mode', and one column per fit parameter.
    The row order matches the input list order.
    """
    return pd.DataFrame([r.to_dict() for r in results])


# ============================================================
# ΔG(r) fitting API
# ============================================================

def build_reference(
    reference_gr_paths: Union[str, Path, List[Union[str, Path]]],
    config: FitConfig,
    verbose: bool = True,
    save_averaged: bool = True,
) -> Reference:
    """
    Build a Reference for ΔG fitting from one or more pre-t0 .gr files.

    Workflow
    --------
    1. Load every .gr file with PDFParser.
    2. Interpolate onto the common r-grid (config.rmin → rmax, step rstep).
    3. Average G(r) across files.
    4. Write the averaged G(r) to disk so PDFParser can read it back.
    5. Full-fit the averaged G(r) → reference params.

    Parameters
    ----------
    reference_gr_paths : path, list of paths, or glob
        Pre-t0 reference files. A single path is treated as an average of one.
    config : FitConfig
    verbose : bool
    save_averaged : bool
        If True and config.save_outputs, save the averaged G(r) as
        <resdir>/reference_avg.gr.

    Returns
    -------
    Reference
    """
    paths = _resolve_reference_paths(reference_gr_paths)
    if not paths:
        raise FileNotFoundError(f"No .gr files matched: {reference_gr_paths!r}")

    if verbose:
        print(f"Building reference from {len(paths)} file(s):")
        for p in paths:
            print(f"  {p.name}")

    # ---- Load + interpolate onto common grid ----
    r_common = np.arange(config.rmin, config.rmax + 0.5 * config.rstep, config.rstep)
    g_stack  = np.zeros((len(paths), len(r_common)), dtype=float)

    for i, p in enumerate(paths):
        r_i, g_i = _read_gr_arrays(p)
        if (len(r_i) != len(r_common)
            or not np.allclose(r_i, r_common, atol=1e-6)):
            g_i = np.interp(r_common, r_i, g_i)
        g_stack[i] = g_i

    g_avg = g_stack.mean(axis=0)

    # ---- Write averaged G(r) so PDFParser can read it for the reference fit ----
    if save_averaged and config.save_outputs:
        ref_dir = _resolve_ref_outdir(config, paths[0])
        ref_dir.mkdir(parents=True, exist_ok=True)
        avg_path = ref_dir / "reference_avg.gr"
        _write_gr_file(avg_path, r_common, g_avg,
                       header=f"Averaged reference from {len(paths)} file(s)")
        if verbose:
            print(f"  Averaged G(r) written to {avg_path}")
    else:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".gr", delete=False)
        tmp.close()
        avg_path = Path(tmp.name)
        _write_gr_file(avg_path, r_common, g_avg, header="Averaged reference (tmp)")

    # ---- Full-fit the averaged reference ----
    if verbose:
        print("Full-fitting averaged reference G(r)...")
    ref_result = fit_one(avg_path, config, mode="full", verbose=verbose)
    if verbose:
        print(f"  Reference Rw = {ref_result.rw:.4f}")

    return Reference(
        r=r_common,
        g=g_avg,
        params=ref_result.params,
        rw=ref_result.rw,
        source_paths=paths,
        recipe=ref_result.recipe,
    )


def fit_dgr(
    gr_path: Union[str, Path],
    reference: Reference,
    config: FitConfig,
    initial_params: Optional[Dict] = None,
    pin_scale: bool = True,
    verbose: bool = True,
) -> FitResult:
    """
    Fit ΔG(r) = G(r,t) - G_ref(r) for a single time-resolved file.

    Architecture
    ------------
    Builds a two-generator FitContribution where:
      G1   = perturbed-state model (structure params refined)
      Gref = reference-state model (all params frozen at reference.params)
    The equation is "s1*G1 - s1_ref*Gref", fit against ΔG_data on the
    profile. Qdamp, Qbroad, sinc termination, and the static structure all
    cancel to first order, so the residual is sensitive to small changes.

    Parameters
    ----------
    gr_path : str or Path
    reference : Reference
    config : FitConfig
    initial_params : dict, optional
        Warm-start for the perturbed-state parameters. Defaults to
        reference.params (small-perturbation assumption).
    pin_scale : bool
        If True (default), s1 is held at reference.params['s1']. Recommended
        when pump-probe data normalization is consistent. Set False if you
        genuinely expect a scale change.
    verbose : bool

    Returns
    -------
    FitResult
        params: absolute perturbed-state values. Compute deltas as
                params - reference.params downstream.
        rw:     ΔG-fit Rw; NOT directly comparable to absolute-G(r) Rw because
                the denominator is sum(ΔG²), not sum(G²).
        mode:   'dgr'.
    """
    gr_path = Path(gr_path)

    if verbose:
        print(f"Fitting ΔG(r) for {gr_path.name}")

    recipe = _make_dgr_recipe(config, gr_path, reference,
                              initial_params=initial_params,
                              pin_scale=pin_scale)
    recipe.fithooks[0].verbose = 0
    _fit_dgr(recipe, config, verbose=verbose)

    res = FitResults(recipe)

    if config.save_outputs:
        _save_dgr_outputs(recipe, res, gr_path, config)

    # Only the perturbed-state params go in the result dict (drop *_ref).
    params = {name: recipe.get(name).value
              for name in recipe._parameters.keys()
              if not name.endswith("_ref")}

    return FitResult(params=params, rw=res.rw, mode="dgr",
                     gr_path=gr_path, recipe=recipe)


def fit_dgr_batch(
    gr_paths: Union[str, Path, List[Union[str, Path]]],
    reference: Union[Reference, str, Path, List[Union[str, Path]]],
    config: FitConfig,
    pin_scale: bool = True,
    exclude_reference_files: bool = True,
) -> List[FitResult]:
    """
    Batch ΔG fit against a single Reference.

    Parameters
    ----------
    gr_paths : str, Path, or list
        Glob/dir/list for the time-resolved series.
    reference : Reference, path, or list of paths
        Prebuilt Reference, or paths to pre-t0 files (build_reference is
        called internally in that case).
    config : FitConfig
    pin_scale : bool
        Passed to fit_dgr.
    exclude_reference_files : bool
        If True, skip files used to build the reference (their ΔG ≈ 0).

    Returns
    -------
    list of FitResult in input file order.
    """
    files = _resolve_gr_paths(gr_paths)
    if not files:
        raise FileNotFoundError(f"No .gr files found for: {gr_paths!r}")

    if not isinstance(reference, Reference):
        reference = build_reference(reference, config, verbose=True)

    if exclude_reference_files:
        ref_names = {p.name for p in reference.source_paths}
        kept = [f for f in files if f.name not in ref_names]
        skipped = len(files) - len(kept)
        if skipped:
            print(f"Excluded {skipped} reference file(s) from batch.")
        files = kept

    print(f"\n=== ΔG fits: {len(files)} file(s) ===")
    results: List[FitResult] = []
    params = dict(reference.params)
    for idx, gr_path in enumerate(files, 1):
        print(f"  [{idx}/{len(files)}] {gr_path.name}...", end=" ", flush=True)
        try:
            result = fit_dgr(gr_path, reference, config,
                             initial_params=params,
                             pin_scale=pin_scale, verbose=False)
            params = result.params
            results.append(result)
            print(f"Rw = {result.rw:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")

    return results


# ============================================================
# Visualization
# ============================================================

def plot_series(
    source: Union[pd.DataFrame, str, Path],
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> "plt.Figure":
    """
    Multi-panel plot of all fit parameters vs. file index.

    Parameters
    ----------
    source : DataFrame, CSV path, or str
        Output of results_to_dataframe() or a path to a saved CSV.
    show : bool
        Call plt.show() after building the figure.
    save_path : Path or None
        If given, save the figure as a PDF at this path.

    Returns
    -------
    fig : matplotlib Figure
    """
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source)
    else:
        df = source.copy()

    if len(df) == 0:
        print("plot_series: empty DataFrame — nothing to plot.")
        return None

    if "index" in df.columns:
        df = df.sort_values("index").reset_index(drop=True)
        indices = df["index"].values
    else:
        indices = np.arange(1, len(df) + 1)

    _lat_names  = {"a", "b", "c", "alpha", "beta", "gamma"}
    lat_cols    = [c for c in df.columns if c in _lat_names]
    adp_cols    = [c for c in df.columns if c.startswith("ADP_")]
    fixed_order = ["rw", "s1", "Delta2", "Calib_Qdamp", "Calib_Qbroad"]
    plot_params = lat_cols + [p for p in fixed_order if p in df.columns] + adp_cols

    if not plot_params:
        print("plot_series: no plottable columns found.")
        return None

    param_labels = {
        "rw":           "Rw",
        "a":            "a (Å)",
        "b":            "b (Å)",
        "c":            "c (Å)",
        "alpha":        "α (°)",
        "beta":         "β (°)",
        "gamma":        "γ (°)",
        "s1":           "Scale",
        "Delta2":       "δ₂ (Å²)",
        "Calib_Qdamp":  "Qdamp (Å⁻¹)",
        "Calib_Qbroad": "Qbroad (Å⁻¹)",
    }

    has_mode = "mode" in df.columns
    if has_mode:
        full_mask    = (df["mode"] == "full").values
        reduced_mask = (df["mode"] == "reduced").values
        dgr_mask     = (df["mode"] == "dgr").values

    ncols = 2
    nrows = int(np.ceil(len(plot_params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3 * nrows))
    axes = axes.flatten()

    for ax, param in zip(axes, plot_params):
        label = param_labels.get(param, param)
        if has_mode:
            if full_mask.any():
                ax.plot(indices[full_mask],    df.loc[full_mask,    param],
                        marker="o", ms=5, lw=0,                label="full")
            if reduced_mask.any():
                ax.plot(indices[reduced_mask], df.loc[reduced_mask, param],
                        marker="o", ms=4, lw=1.0, mfc="none",  label="reduced")
            if dgr_mask.any():
                ax.plot(indices[dgr_mask],     df.loc[dgr_mask,     param],
                        marker="s", ms=4, lw=1.0, mfc="none",  label="ΔG")
        else:
            ax.plot(indices, df[param], marker="o", ms=4, lw=1.2)
        ax.set_xlabel("File index")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.tick_params(axis="both", which="major", top=True, right=True)
        ax.ticklabel_format(useOffset=False, axis="y")

    if has_mode and len(plot_params) > 0:
        axes[0].legend(loc="best", fontsize=8)

    for ax in axes[len(plot_params):]:
        ax.set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="pdf")
        print(f"Series plot saved to {save_path}")

    if not show:
        plt.close(fig)

    return None


# ============================================================
# Utilities (also useful directly in notebooks)
# ============================================================

def normalize_full_indices(full_indices, n_files: int) -> set:
    """
    Convert a full_indices spec into a set of 1-based integer file indices.

    Accepts:
        (start, end)          -- inclusive range
        [(start, end), ...]   -- multiple ranges
        [i, j, k, ...]        -- explicit indices
        mixed list            -- ranges and explicit indices combined
    """
    full_set: set = set()

    if (isinstance(full_indices, tuple) and len(full_indices) == 2
            and all(isinstance(x, int) for x in full_indices)):
        start, end = full_indices
        full_set.update(range(max(1, start), min(n_files, end) + 1))
    elif isinstance(full_indices, (list, set)):
        for item in full_indices:
            if isinstance(item, tuple) and len(item) == 2:
                start, end = item
                full_set.update(range(max(1, start), min(n_files, end) + 1))
            elif isinstance(item, int):
                if 1 <= item <= n_files:
                    full_set.add(item)
            else:
                raise ValueError(f"Unrecognized entry in full_indices: {item!r}")
    else:
        raise ValueError(f"Unrecognized full_indices format: {full_indices!r}")

    return full_set


def average_reference_params(full_records: List[Dict]) -> Dict:
    """
    Average numeric parameters across a list of full-mode result dicts.

    Non-numeric columns ('file', 'mode', 'index') are skipped.
    """
    if not full_records:
        return {}
    df = pd.DataFrame(full_records)
    avg = {}
    for col in df.columns:
        if col in ("file", "mode", "index"):
            continue
        try:
            avg[col] = float(df[col].mean())
        except (TypeError, ValueError):
            continue
    return avg


def save_reference_params(params: Dict, path: Union[str, Path]) -> None:
    """Write a parameter dict to JSON (converts numpy scalars to float)."""
    serializable = {k: float(v) for k, v in params.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Reference parameters saved to {path}")


# ============================================================
# Private helpers
# ============================================================

def _resolve_gr_paths(gr_paths) -> List[Path]:
    """Accept a glob string, a directory, or a list of paths; return sorted list."""
    if isinstance(gr_paths, (list, tuple)):
        return sorted(Path(p) for p in gr_paths)
    gr_paths = Path(gr_paths)
    if gr_paths.is_dir():
        return sorted(gr_paths.glob("*.gr"))
    return sorted(gr_paths.parent.glob(gr_paths.name))


def _resolve_reference_paths(spec) -> List[Path]:
    """Accept a path, glob, or list; return sorted list of existing .gr files."""
    if isinstance(spec, (list, tuple)):
        paths = [Path(p) for p in spec]
    else:
        spec = Path(spec)
        if spec.is_dir():
            paths = sorted(spec.glob("*.gr"))
        elif any(ch in str(spec) for ch in "*?["):
            paths = sorted(spec.parent.glob(spec.name))
        else:
            paths = [spec]
    return [p for p in paths if p.exists()]


def _resolve_ref_outdir(config: FitConfig, fallback: Path) -> Path:
    if config.resdir is not None:
        return Path(config.resdir)
    return fallback.parent / "res"


def _read_gr_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Use PDFParser to read a .gr file and return (r, g) arrays."""
    parser = PDFParser()
    parser.parseFile(str(path))
    # PDFParser exposes x/y via .x / .y after parseFile (srfit convention),
    # but the more reliable way is to feed it through a Profile.
    profile = Profile()
    profile.loadParsedData(parser)
    return np.asarray(profile.xobs), np.asarray(profile.yobs)


def _write_gr_file(path: Path, r: np.ndarray, g: np.ndarray, header: str = "") -> None:
    """
    Write a minimal PDFgetX-style .gr file that PDFParser can read.
    """
    with open(path, "w") as f:
        if header:
            for line in header.splitlines():
                f.write(f"# {line}\n")
        f.write("##### start data\n")
        f.write("#L r(A)  G(r)\n")
        for ri, gi in zip(r, g):
            f.write(f"{ri:.6f}  {gi:.6e}\n")


def _make_recipe(
    config: FitConfig,
    dat_path: Path,
    initial_params: Optional[Dict] = None,
    fix_instrumental: bool = False,
) -> FitRecipe:
    if initial_params is None:
        initial_params = {}

    p_cif = get_parser("cif")
    stru = p_cif.parse_file(str(config.cif_path))

    profile = Profile()
    parser  = PDFParser()
    parser.parseFile(str(dat_path))
    profile.loadParsedData(parser)
    profile.setCalculationRange(xmin=config.rmin, xmax=config.rmax, dx=config.rstep)

    generator = PDFGenerator("G1")
    generator.setStructure(stru, periodic=True)

    if config.run_parallel:
        try:
            import psutil
            import multiprocessing
            from multiprocessing import Pool
            n_cores    = multiprocessing.cpu_count()
            avail      = np.floor((100 - psutil.cpu_percent()) / (100.0 / n_cores))
            ncpu       = int(max(1, avail))
            pool       = Pool(processes=ncpu)
            generator.parallel(ncpu=ncpu, mapfunc=pool.map)
        except ImportError:
            print("Parallelization packages unavailable; running serially.")

    contribution = FitContribution("crystal")
    contribution.addProfileGenerator(generator)
    contribution.setProfile(profile, xname="r")
    contribution.setEquation("s1*G1")

    recipe = FitRecipe()
    recipe.addContribution(contribution)

    recipe.crystal.G1.qdamp.value  = initial_params.get("Calib_Qdamp",  config.qdamp_i)
    recipe.crystal.G1.qbroad.value = initial_params.get("Calib_Qbroad", config.qbroad_i)
    recipe.crystal.G1.setQmax(config.qmax)
    recipe.crystal.G1.setQmin(config.qmin)

    recipe.addVar(contribution.s1,
                  initial_params.get("s1", config.scale_i),
                  tag="scale")

    sgparams = constrainAsSpaceGroup(generator.phase, config.space_group)
    for par in sgparams.latpars:
        recipe.addVar(par,
                      value=initial_params.get(par.name, par.value),
                      fixed=False, name=par.name, tag="lat")
    for par in sgparams.adppars:
        adp_name = f"ADP_{par.name}"
        recipe.addVar(par,
                      value=initial_params.get(adp_name, config.uiso_i),
                      fixed=False, name=adp_name, tag="adp")

    recipe.addVar(generator.delta2,
                  name="Delta2",
                  value=initial_params.get("Delta2", config.delta2_i),
                  tag="d2")

    qdamp_fixed  = fix_instrumental or config.fix_qdamp
    qbroad_fixed = fix_instrumental or config.fix_qbroad
    recipe.addVar(generator.qdamp,
                  fixed=qdamp_fixed, name="Calib_Qdamp",
                  value=initial_params.get("Calib_Qdamp", config.qdamp_i),
                  tag="inst")
    recipe.addVar(generator.qbroad,
                  fixed=qbroad_fixed, name="Calib_Qbroad",
                  value=initial_params.get("Calib_Qbroad", config.qbroad_i),
                  tag="inst")

    return recipe


def _make_dgr_recipe(
    config: FitConfig,
    gr_path: Path,
    reference: Reference,
    initial_params: Optional[Dict] = None,
    pin_scale: bool = True,
) -> FitRecipe:
    """
    Build a FitRecipe with two generators (perturbed + reference) for ΔG fits.
    """
    if initial_params is None:
        initial_params = dict(reference.params)

    # ---- Parse structure twice — independent copies for the two generators ----
    p_cif = get_parser("cif")
    stru_pert = p_cif.parse_file(str(config.cif_path))
    stru_ref  = p_cif.parse_file(str(config.cif_path))

    # ---- Load time-resolved data, then replace y with ΔG_data ----
    profile = Profile()
    parser  = PDFParser()
    parser.parseFile(str(gr_path))
    profile.loadParsedData(parser)
    profile.setCalculationRange(xmin=config.rmin, xmax=config.rmax, dx=config.rstep)

    r_prof = np.asarray(profile.x)
    g_prof = np.asarray(profile.y)
    g_ref_on_prof = np.interp(r_prof, reference.r, reference.g)
    dg_data = g_prof - g_ref_on_prof

    # Use dy from the file if available, else 1.0 weights.
    dy = getattr(profile, "dy", None)
    if dy is None or np.all(np.asarray(dy) == 0):
        dy = np.ones_like(dg_data)

    # setObservedProfile signature varies across srfit versions:
    # some accept (xobs, yobs, dyobs) positionally, some accept dy= kwarg,
    # and some only take (xobs, yobs). Try in order of preference.
    try:
        profile.setObservedProfile(r_prof, dg_data, dy)
    except TypeError:
        try:
            profile.setObservedProfile(r_prof, dg_data)
        except TypeError:
            # Fallback: assign attributes directly.
            profile.xobs  = r_prof
            profile.yobs  = dg_data
            profile.dyobs = dy
            profile.x = r_prof
            profile.y = dg_data
            profile.dy = dy

    profile.setCalculationRange(xmin=config.rmin, xmax=config.rmax, dx=config.rstep)

    # ---- Perturbed generator G1 ----
    gen_pert = PDFGenerator("G1")
    gen_pert.setStructure(stru_pert, periodic=True)
    if config.run_parallel:
        try:
            import psutil, multiprocessing
            from multiprocessing import Pool
            n_cores = multiprocessing.cpu_count()
            avail   = np.floor((100 - psutil.cpu_percent()) / (100.0 / n_cores))
            ncpu    = int(max(1, avail))
            pool    = Pool(processes=ncpu)
            gen_pert.parallel(ncpu=ncpu, mapfunc=pool.map)
        except ImportError:
            pass

    # ---- Reference generator Gref ----
    gen_ref = PDFGenerator("Gref")
    gen_ref.setStructure(stru_ref, periodic=True)
    # Not parallelized: parameters never change so the result is identical
    # every call. (If profiling shows this dominates, cache the array and
    # subtract numerically instead of using a generator at all.)

    contribution = FitContribution("crystal")
    contribution.addProfileGenerator(gen_pert)
    contribution.addProfileGenerator(gen_ref)
    contribution.setProfile(profile, xname="r")
    contribution.setEquation("s1*G1 - s1_ref*Gref")

    recipe = FitRecipe()
    recipe.addContribution(contribution)

    # ---- Instrumental params: same Q-range for both ----
    s_ref_val = reference.params.get("s1", config.scale_i)
    for gen in (gen_pert, gen_ref):
        gen.qdamp.value  = reference.params.get("Calib_Qdamp",  config.qdamp_i)
        gen.qbroad.value = reference.params.get("Calib_Qbroad", config.qbroad_i)
        gen.setQmax(config.qmax)
        gen.setQmin(config.qmin)

    # ---- Scales: s1 (free or pinned), s1_ref (always pinned) ----
    recipe.addVar(contribution.s1,
                  value=initial_params.get("s1", s_ref_val),
                  tag="scale")
    # setEquation("... s1_ref*Gref") above causes srfit to auto-create s1_ref
    # on the contribution. If for some reason it didn't, create it manually.
    # Either way, do NOT register it as a recipe Var (never refined).
    if not hasattr(contribution, "s1_ref"):
        contribution.newParameter("s1_ref", value=s_ref_val)
    contribution.s1_ref.value = s_ref_val
    if pin_scale:
        recipe.get("s1").value = s_ref_val
        recipe.fix("s1")

    # ---- Perturbed-structure space-group constraints (free) ----
    sgparams_pert = constrainAsSpaceGroup(gen_pert.phase, config.space_group)
    for par in sgparams_pert.latpars:
        recipe.addVar(par,
                      value=initial_params.get(par.name, par.value),
                      fixed=False, name=par.name, tag="lat")
    for par in sgparams_pert.adppars:
        adp_name = f"ADP_{par.name}"
        recipe.addVar(par,
                      value=initial_params.get(adp_name, config.uiso_i),
                      fixed=False, name=adp_name, tag="adp")

    # ---- Reference-structure space-group constraints (all fixed) ----
    sgparams_ref = constrainAsSpaceGroup(gen_ref.phase, config.space_group)
    for par in sgparams_ref.latpars:
        ref_val = reference.params.get(par.name, par.value)
        recipe.addVar(par, value=ref_val, fixed=True,
                      name=f"{par.name}_ref", tag="lat_ref")
    for par in sgparams_ref.adppars:
        adp_name = f"ADP_{par.name}"
        ref_val = reference.params.get(adp_name, config.uiso_i)
        recipe.addVar(par, value=ref_val, fixed=True,
                      name=f"{adp_name}_ref", tag="adp_ref")

    # ---- Delta2 (pinned in both generators) ----
    recipe.addVar(gen_pert.delta2,
                  name="Delta2",
                  value=reference.params.get("Delta2", config.delta2_i),
                  fixed=True, tag="d2")
    recipe.addVar(gen_ref.delta2,
                  name="Delta2_ref",
                  value=reference.params.get("Delta2", config.delta2_i),
                  fixed=True, tag="d2_ref")

    # ---- Instrumental params (pinned, both generators) ----
    recipe.addVar(gen_pert.qdamp,
                  name="Calib_Qdamp",
                  value=reference.params.get("Calib_Qdamp", config.qdamp_i),
                  fixed=True, tag="inst")
    recipe.addVar(gen_pert.qbroad,
                  name="Calib_Qbroad",
                  value=reference.params.get("Calib_Qbroad", config.qbroad_i),
                  fixed=True, tag="inst")
    recipe.addVar(gen_ref.qdamp,
                  name="Calib_Qdamp_ref",
                  value=reference.params.get("Calib_Qdamp", config.qdamp_i),
                  fixed=True, tag="inst_ref")
    recipe.addVar(gen_ref.qbroad,
                  name="Calib_Qbroad_ref",
                  value=reference.params.get("Calib_Qbroad", config.qbroad_i),
                  fixed=True, tag="inst_ref")

    return recipe


def _get_bounds(recipe: FitRecipe, config: FitConfig):
    """Build (lower, upper) bound arrays for currently free params; clips x0."""
    names = recipe.names
    if not names:
        return None

    lower = np.full(len(names), -np.inf)
    upper = np.full(len(names),  np.inf)

    for i, name in enumerate(names):
        # ΔG fits create *_ref names — strip the suffix when looking up bounds.
        lookup = name[:-4] if name.endswith("_ref") else name
        if lookup in config.param_bounds:
            lo, hi = config.param_bounds[lookup]
            if lo is not None: lower[i] = lo
            if hi is not None: upper[i] = hi
        elif lookup.startswith("ADP_") and "_adp_default" in config.param_bounds:
            lo, hi = config.param_bounds["_adp_default"]
            if lo is not None: lower[i] = lo
            if hi is not None: upper[i] = hi

    eps  = 1e-9
    vals = np.array(recipe.values, dtype=float)
    vals = np.minimum(np.maximum(vals, lower + eps), upper - eps)
    for i, name in enumerate(names):
        recipe.get(name).value = vals[i]

    return (lower, upper)


def _run_least_squares(recipe: FitRecipe, config: FitConfig):
    bounds = _get_bounds(recipe, config)
    if bounds is None:
        return None
    return least_squares(recipe.residual, recipe.values,
                         method="trf", x_scale="jac", bounds=bounds)


def _rw(recipe: FitRecipe) -> float:
    """Compute current weighted residual directly from the profile arrays."""
    y     = recipe.crystal.profile.y
    ycalc = recipe.crystal.profile.ycalc
    return float(np.sqrt(np.sum((y - ycalc) ** 2) / np.sum(y ** 2)))


def _fit_full(recipe: FitRecipe, config: FitConfig, verbose: bool = False) -> None:
    recipe.fix("all")

    recipe.free("scale")
    _run_least_squares(recipe, config)
    if verbose: print(f"    scale          Rw = {_rw(recipe):.4f}")

    recipe.free("lat")
    _run_least_squares(recipe, config)
    if verbose: print(f"    + lat          Rw = {_rw(recipe):.4f}")

    recipe.free("adp")
    _run_least_squares(recipe, config)
    if verbose: print(f"    + adp          Rw = {_rw(recipe):.4f}")

    recipe.free("d2")
    _run_least_squares(recipe, config)
    if verbose: print(f"    + Delta2       Rw = {_rw(recipe):.4f}")

    if not (config.fix_qdamp and config.fix_qbroad):
        recipe.free("inst")
        _run_least_squares(recipe, config)
        if verbose: print(f"    + inst         Rw = {_rw(recipe):.4f}")

    _run_least_squares(recipe, config)
    if verbose: print(f"    final polish   Rw = {_rw(recipe):.4f}")


def _fit_reduced(recipe: FitRecipe, config: FitConfig, verbose: bool = False) -> None:
    recipe.fix("all")

    recipe.free("scale")
    _run_least_squares(recipe, config)
    if verbose: print(f"    scale          Rw = {_rw(recipe):.4f}")

    recipe.free("lat")
    _run_least_squares(recipe, config)
    if verbose: print(f"    + lat          Rw = {_rw(recipe):.4f}")

    recipe.free("adp")
    _run_least_squares(recipe, config)
    if verbose: print(f"    + adp          Rw = {_rw(recipe):.4f}")

    _run_least_squares(recipe, config)
    if verbose: print(f"    final polish   Rw = {_rw(recipe):.4f}")


def _fit_dgr(recipe: FitRecipe, config: FitConfig, verbose: bool = False) -> None:
    """Staged ΔG refinement: (optional scale) → lat → adp → polish."""
    recipe.fix("all")

    if "s1" in recipe.names:
        # only free if it wasn't pinned in _make_dgr_recipe
        if not recipe.get("s1").const:
            recipe.free("s1")
            _run_least_squares(recipe, config)
            if verbose: print(f"    scale          Rw = {_rw(recipe):.4f}")

    recipe.free("lat")
    _run_least_squares(recipe, config)
    if verbose: print(f"    + lat          Rw = {_rw(recipe):.4f}")

    recipe.free("adp")
    _run_least_squares(recipe, config)
    if verbose: print(f"    + adp          Rw = {_rw(recipe):.4f}")

    _run_least_squares(recipe, config)
    if verbose: print(f"    final polish   Rw = {_rw(recipe):.4f}")


def _save_fit_outputs(
    recipe: FitRecipe,
    res: FitResults,
    gr_path: Path,
    config: FitConfig,
) -> None:
    basename = gr_path.stem
    resdir = Path(config.resdir) if config.resdir else gr_path.parent / "res"
    fitdir = Path(config.fitdir) if config.fitdir else gr_path.parent / "fit"
    figdir = Path(config.figdir) if config.figdir else gr_path.parent / "fig"
    for d in (resdir, fitdir, figdir):
        d.mkdir(exist_ok=True)

    recipe.crystal.profile.savetxt(fitdir / f"{basename}.fit")
    res.saveResults(resdir / f"{basename}.res", header=f"{basename}\n")
    _plot_fit(recipe, show=False, save_path=figdir / f"{basename}.pdf")


def _save_dgr_outputs(
    recipe: FitRecipe,
    res: FitResults,
    gr_path: Path,
    config: FitConfig,
) -> None:
    basename = gr_path.stem + "_dgr"
    resdir = Path(config.resdir) if config.resdir else gr_path.parent / "res"
    fitdir = Path(config.fitdir) if config.fitdir else gr_path.parent / "fit"
    figdir = Path(config.figdir) if config.figdir else gr_path.parent / "fig"
    for d in (resdir, fitdir, figdir):
        d.mkdir(exist_ok=True)

    recipe.crystal.profile.savetxt(fitdir / f"{basename}.fit")
    res.saveResults(resdir / f"{basename}.res", header=f"{basename}\n")
    _plot_dgr_fit(recipe, show=False, save_path=figdir / f"{basename}.pdf")


def _plot_fit(
    recipe: FitRecipe,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> "plt.Figure":
    r      = recipe.crystal.profile.x
    g      = recipe.crystal.profile.y
    gcalc  = recipe.crystal.profile.ycalc
    diffzero = -0.65 * max(g) * np.ones_like(g)
    diff     = g - gcalc + diffzero

    mpl.rcParams.update(mpl.rcParamsDefault)
    mplstyle = Path(__file__).parent / "utils" / "billinge.mplstyle"
    if mplstyle.exists():
        plt.style.use(str(mplstyle))

    fig, ax = plt.subplots(1, 1)
    ax.plot(r, diffzero, lw=1.0, ls="--", c="black")
    ax.plot(r, g,      ls="None", marker="o", ms=5, mew=0.2, mfc="None", label="G(r) Data")
    ax.plot(r, gcalc,  lw=1.3,  label="G(r) Fit")
    ax.plot(r, diff,   lw=1.2,  label="G(r) diff")
    ax.set_xlabel(r"r ($\mathrm{\AA}$)")
    ax.set_ylabel(r"G ($\mathrm{\AA}$$^{-2}$)")
    ax.tick_params(axis="both", which="major", top=True, right=True)
    ax.set_xlim(r[0], r[-1])
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="pdf")

    if not show:
        plt.close(fig)

    return fig


def _plot_dgr_fit(
    recipe: FitRecipe,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> "plt.Figure":
    """Plot ΔG_data, ΔG_model, and residual."""
    r      = recipe.crystal.profile.x
    dg     = recipe.crystal.profile.y
    dgcalc = recipe.crystal.profile.ycalc
    yrange = max(abs(dg.min()), abs(dg.max()), 1e-12)
    diffzero = -1.5 * yrange * np.ones_like(dg)
    diff     = dg - dgcalc + diffzero

    mpl.rcParams.update(mpl.rcParamsDefault)
    mplstyle = Path(__file__).parent / "utils" / "billinge.mplstyle"
    if mplstyle.exists():
        plt.style.use(str(mplstyle))

    fig, ax = plt.subplots(1, 1)
    ax.axhline(0, lw=0.8, c="gray", alpha=0.5)
    ax.plot(r, diffzero, lw=1.0, ls="--", c="black")
    ax.plot(r, dg,     ls="None", marker="o", ms=4, mew=0.2, mfc="None", label=r"$\Delta$G(r) Data")
    ax.plot(r, dgcalc, lw=1.3,  label=r"$\Delta$G(r) Fit")
    ax.plot(r, diff,   lw=1.0,  label="Residual")
    ax.set_xlabel(r"r ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$\Delta$G ($\mathrm{\AA}$$^{-2}$)")
    ax.tick_params(axis="both", which="major", top=True, right=True)
    ax.set_xlim(r[0], r[-1])
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="pdf")

    if not show:
        plt.close(fig)

    return fig


# ============================================================
# CLI entry point (mirrors the original script behaviour)
# ============================================================

def main():
    """
    CLI wrapper. Edit the variables below and run:
        python diffpy_fitting.py
    For interactive use, import fit_one / fit_batch instead.
    """
    DATA_PATH     = Path(r"C:\Users\lheald\Documents\Guzelturk_Lab\Cotts_Processed_Data\BTO400nmS3_360Kre4\output")
    CIF_NAME      = "BaTiO3_tetragonal.cif"
    GR_PATTERN    = "BaTiO3*.gr"
    FULL_INDICES  = (1, 10)
    FIT_ID        = "Fit_BaTiO3_Bulk"

    cfg = FitConfig(
        cif_path=DATA_PATH / CIF_NAME,
        resdir=DATA_PATH / "res",
        fitdir=DATA_PATH / "fit",
        figdir=DATA_PATH / "fig",
    )

    results = fit_batch(DATA_PATH / GR_PATTERN, cfg, full_indices=FULL_INDICES)

    df = results_to_dataframe(results)
    df.insert(0, "index", range(1, len(df) + 1))

    results_path = DATA_PATH / f"{FIT_ID}_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\nSummary saved to {results_path}")

    plot_series(df, show=False, save_path=DATA_PATH / f"{FIT_ID}_series.pdf")


if __name__ == "__main__":
    main()
