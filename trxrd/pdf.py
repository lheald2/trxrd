import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from globals import FIGSIZE, FORM_FACTOR_FILE


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


def parse_composition_string(formula):
    """
    Parse a chemical formula with integer or decimal stoichiometries.

    Examples
    --------
    BaTiO3 -> {'Ba': 1.0, 'Ti': 1.0, 'O': 3.0}
    Cs0.097Pb1C0.904N1.679I1.798Br1.105Cl0.097
        -> {'Cs': 0.097, 'Pb': 1.0, 'C': 0.904, 'N': 1.679,
            'I': 1.798, 'Br': 1.105, 'Cl': 0.097}
    """
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("Formula must be a non-empty string.")

    formula = formula.strip()
    pattern = r"([A-Z][a-z]?)(\d*\.?\d*)"
    matches = list(re.finditer(pattern, formula))

    if not matches:
        raise ValueError(f"Could not parse composition: {formula}")

    composition_dict = {}
    consumed = ""

    for m in matches:
        elem = m.group(1)
        count = m.group(2)
        count_val = float(count) if count else 1.0
        composition_dict[elem] = composition_dict.get(elem, 0.0) + count_val
        consumed += m.group(0)

    if consumed != formula:
        raise ValueError(f"Could not fully parse composition: {formula}")

    return composition_dict


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
        composition_dict = parse_composition_string(composition)

    elif isinstance(composition, dict):
        composition_dict = {elem: float(count) for elem, count in composition.items()}

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
            _, axes = plt.subplots(1, 2, figsize=figsize)
            ax0, ax1 = axes
        else:
            _, ax0 = plt.subplots(1, 1, figsize=figsize)

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


def fit_iq_to_f2_high_q(
    q,
    iq,
    f2_avg,
    q_fit_range,
    background="constant",
    plot=False,
    figsize=(7, 4),
    return_dict=True,
):
    """
    Fit a correction to I(Q) so that the corrected intensity matches <f^2(Q)>
    over a chosen high-Q region.

    The fitted model is:

        I_corr(Q) = a * I(Q) + b(Q)

    where b(Q) can be:
        - "none"     : 0
        - "constant" : b
        - "linear"   : b + cQ

    Parameters
    ----------
    q : np.ndarray
        1D Q axis, shape (n_q,)
    iq : np.ndarray
        1D or 2D intensity array:
        - (n_q,)
        - (n_profiles, n_q)
    f2_avg : np.ndarray
        1D <f^2(Q)> array, shape (n_q,)
    q_fit_range : tuple
        (q_min, q_max) fit range for matching I(Q) to <f^2(Q)>
    background : {"none", "constant", "linear"}
        Background model to include in addition to the scale factor.
    plot : bool
        If True, plot the fit for one profile.
    figsize : tuple
        Figure size for plotting.
    return_dict : bool
        If True, return a dictionary, else return corrected intensity only.

    Returns
    -------
    result : dict or np.ndarray
        If return_dict=True:
            {
                "iq_corrected": corrected intensity array,
                "fit_mask": boolean mask used for fitting,
                "coefficients": fitted coefficients,
                "background": background model,
                "input_was_1d": bool,
            }

        coefficients are:
            - "none"     : [a]
            - "constant" : [a, b]
            - "linear"   : [a, b, c]
    """
    q = np.asarray(q, dtype=float)
    iq = np.asarray(iq, dtype=float)
    f2_avg = np.asarray(f2_avg, dtype=float)

    if q.ndim != 1:
        raise ValueError("q must be 1D.")
    if f2_avg.ndim != 1:
        raise ValueError("f2_avg must be 1D.")
    if len(q) != len(f2_avg):
        raise ValueError("q and f2_avg must have the same length.")

    if iq.ndim == 1:
        iq_2d = iq[None, :]
        input_was_1d = True
    elif iq.ndim == 2:
        iq_2d = iq
        input_was_1d = False
    else:
        raise ValueError("iq must be 1D or 2D.")

    if iq_2d.shape[1] != len(q):
        raise ValueError("iq.shape[-1] must match len(q).")

    if background not in ("none", "constant", "linear"):
        raise ValueError("background must be 'none', 'constant', or 'linear'.")

    if q_fit_range is None or len(q_fit_range) != 2:
        raise ValueError("q_fit_range must be a tuple: (q_min, q_max).")

    q_min, q_max = q_fit_range
    if q_min >= q_max:
        raise ValueError("q_fit_range must satisfy q_min < q_max.")

    fit_mask = np.isfinite(q) & np.isfinite(f2_avg) & (q >= q_min) & (q <= q_max)
    if np.sum(fit_mask) < 3:
        raise ValueError("Not enough valid points in q_fit_range for fitting.")

    q_fit = q[fit_mask]
    y_target = f2_avg[fit_mask]

    iq_corrected_2d = np.full_like(iq_2d, np.nan, dtype=float)
    coefficients = []

    for i in range(iq_2d.shape[0]):
        y_iq = iq_2d[i, fit_mask]
        finite = np.isfinite(y_iq) & np.isfinite(y_target) & np.isfinite(q_fit)

        if np.sum(finite) < 3:
            coefficients.append(None)
            continue

        x_iq = y_iq[finite]
        x_q = q_fit[finite]
        y = y_target[finite]

        if background == "none":
            A = x_iq[:, None]
        elif background == "constant":
            A = np.column_stack([x_iq, np.ones_like(x_iq)])
        else:  # linear
            A = np.column_stack([x_iq, np.ones_like(x_iq), x_q])

        coeff = np.linalg.lstsq(A, y, rcond=None)[0]
        coefficients.append(coeff)

        if background == "none":
            a = coeff[0]
            iq_corrected_2d[i] = a * iq_2d[i]
        elif background == "constant":
            a, b = coeff
            iq_corrected_2d[i] = a * iq_2d[i] + b
        else:
            a, b, c = coeff
            iq_corrected_2d[i] = a * iq_2d[i] + b + c * q

    if plot:
        idx = 0
        coeff = coefficients[idx]
        if coeff is not None:
            _, ax = plt.subplots(figsize=figsize)

            ax.plot(q, iq_2d[idx], label="Original I(Q)", alpha=0.6)
            ax.plot(q, iq_corrected_2d[idx], label="Corrected I(Q)")
            ax.plot(q, f2_avg, label=r"$\langle f^2(Q)\rangle$", linestyle="--")

            ax.axvspan(q_min, q_max, alpha=0.15, label="Fit range")
            ax.set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)")
            ax.set_ylabel("Intensity")
            ax.set_title("High-Q Scale Fit")
            ax.legend()
            plt.tight_layout()
            plt.show()

    if input_was_1d:
        iq_corrected = iq_corrected_2d[0]
    else:
        iq_corrected = iq_corrected_2d

    if return_dict:
        return {
            "iq_corrected": iq_corrected,
            "fit_mask": fit_mask,
            "coefficients": coefficients[0] if input_was_1d else coefficients,
            "background": background,
            "input_was_1d": input_was_1d,
        }

    return iq_corrected


def correct_iq(
    q,
    iq,
    composition,
    q_fit_range,
    background="constant",
    plot=False,
    return_dict=True,
):
    """
    Empirically correct I(Q) so that high-Q behavior matches <f^2(Q)>.
    """
    q = np.asarray(q, dtype=float)
    iq = np.asarray(iq, dtype=float)

    ff_result = compute_average_form_factors(
        q=q,
        composition=composition,
        plot=False,
        return_dict=True,
    )

    f2_avg = np.asarray(ff_result["f2_avg"], dtype=float)

    fit_result = fit_iq_to_f2_high_q(
        q=q,
        iq=iq,
        f2_avg=f2_avg,
        q_fit_range=q_fit_range,
        background=background,
        plot=plot,
        return_dict=True,
    )

    if return_dict:
        return {
            "q": q,
            "iq_corrected": fit_result["iq_corrected"],
            "coefficients": fit_result["coefficients"],
            "fit_mask": fit_result["fit_mask"],
            "background": fit_result["background"],
            "f2_avg": f2_avg,
            "composition_dict": ff_result["composition_dict"],
            "atomic_fractions": ff_result["atomic_fractions"],
            "input_was_1d": fit_result["input_was_1d"],
        }

    return fit_result["iq_corrected"]


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

        _, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)

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

        _, axes = plt.subplots(1, 2, figsize=figsize)

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


def compute_qualitative_difference_pdf(
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

        _, axes = plt.subplots(1, 2, figsize=figsize)

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
