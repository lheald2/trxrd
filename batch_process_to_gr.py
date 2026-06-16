"""
Reads a series of I(Q) .dat files matching a scan name, converts each to G(r)
(and optionally F(Q), S(Q)) using diffpy.pdfgetx, and saves the results in the
output folder.

Usage: edit the parameters below and run as a script.
"""

from pathlib import Path
import numpy as np
from diffpy.pdfgetx import PDFGetter

# ============================================================
# Paths and scan identification
# ============================================================
DAT_PATH   = Path(r"C:\Users\lheald\Documents\Guzelturk_Lab\Cotts_Processed_Data\BTO_18keV\BTO400nmS3_360Klong")  # Path to directory containing .dat files
SCAN_NAME  = "BTO400nmS3_360Klong"   # filename prefix; set None to load all .dat files
SAVE_PATH  = Path(r"C:\Users\lheald\Documents\Guzelturk_Lab\Cotts_Processed_Data\BTO_18keV\BTO400nmS3_360Klong\output")  # Path to directory where output files will be saved

# Optional: path to a pdfgetx3.cfg file produced by PDFgetX3.
# When provided, all PDF parameters below are ignored.
CFG_FILE   = None   # e.g. Path("pdfgetx3.cfg")

# ============================================================
# PDF calculation parameters (used only when CFG_FILE is None)
# ============================================================
COMPOSITION = "BaTiO3"
WAVELENGTH  = 0.6888  # Å, X-ray wavelength
MODE        = "xray"   # "xray" or "neutron"
DATAFORMAT  = "QA"   # input data format: "QA" (Å⁻¹), "Qnm" (nm⁻¹), "twotheta", "d"
Q_MIN       = 0.35     # Å⁻¹, lower Q cutoff
Q_MAX       = 9.0     # Å⁻¹, upper Q cutoff for PDF transform
QMAXINST    = 9.5     # Å⁻¹, instrument Qmax (sets upper physical limit before cutoff)
R_MIN       = 0.0      # Å, lower r limit
R_MAX       = 30.0     # Å, upper r limit
R_STEP      = 0.01     # Å, r grid step
RPOLY       = 1.5      # Å, degree of r-polynomial correction
WINDOW      = "lorch"  # window function: "lorch", "hanning", "blackman", or None
OUTPUTTYPES = ["gr"]   # any subset of ["gr", "fq", "sq", "iq"]

# ============================================================
# Background subtraction (optional)
# ============================================================
BGFILE  = f"{DAT_PATH}/BaTiO3_background.dat"   # Path to background .dat file, or None
BGSCALE = 6.0 # scale factor applied to background before subtraction

# ============================================================
# Helper: read a .dat file, skipping non-numeric lines
# ============================================================
def read_dat(file_path):
    rows = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            if len(vals) >= 2:
                rows.append(vals[:2])
    if not rows:
        return None, None
    arr = np.array(rows, dtype=float)
    q, iq = arr[:, 0], arr[:, 1]
    valid = np.isfinite(q) & np.isfinite(iq)
    if not np.any(valid):
        return None, None
    return q[valid], iq[valid]

# ============================================================
# Find .dat files
# ============================================================
if not DAT_PATH.exists():
    raise ValueError(f"Folder does not exist: {DAT_PATH}")
if not DAT_PATH.is_dir():
    raise ValueError(f"Path is not a directory: {DAT_PATH}")

pattern = "*.dat" if SCAN_NAME is None else f"{SCAN_NAME}*.dat"
file_names = sorted(DAT_PATH.glob(pattern))
print(f"{len(file_names)} .dat files found in {DAT_PATH}.")

if len(file_names) == 0:
    raise ValueError(f"No .dat files found matching '{pattern}' in {DAT_PATH}")

# ============================================================
# Load background once before the loop
# ============================================================
q_bg, iq_bg = None, None
if BGFILE is not None:
    q_bg, iq_bg = read_dat(BGFILE)
    if q_bg is None:
        raise ValueError(f"No valid data found in background file: {BGFILE}")
    print(f"Background loaded: {Path(BGFILE).name}")

# ============================================================
# Build PDFGetter
# ============================================================
if CFG_FILE is not None:
    from diffpy.pdfgetx import loadPDFConfig
    pg = PDFGetter(config=loadPDFConfig(str(CFG_FILE)))
else:
    pg = PDFGetter()
    pg.config.composition = COMPOSITION
    pg.config.wavelength  = WAVELENGTH
    pg.config.mode        = MODE
    pg.config.dataformat  = DATAFORMAT
    pg.config.qmin        = Q_MIN
    pg.config.qmax        = Q_MAX
    pg.config.qmaxinst    = QMAXINST
    pg.config.rmin        = R_MIN
    pg.config.rmax        = R_MAX
    pg.config.rstep       = R_STEP
    pg.config.rpoly       = RPOLY
    pg.config.window      = WINDOW
    pg.config.outputtypes = OUTPUTTYPES

# ============================================================
# Output type → file extension and column header
# ============================================================
OUTPUT_INFO = {
    "gr": (".gr", "r\tG(r)"),
    "fq": (".fq", "Q\tF(Q)"),
    "sq": (".sq", "Q\tS(Q)"),
    "iq": (".iq", "Q\tI(Q)"),
}

# ============================================================
# Process and save
# ============================================================
SAVE_PATH.mkdir(parents=True, exist_ok=True)

for file_path in file_names:
    q, iq = read_dat(file_path)
    if q is None:
        print(f"Skipping {file_path.name}: no numeric data found.")
        continue

    if q_bg is not None:
        iq = iq - BGSCALE * np.interp(q, q_bg, iq_bg, left=0.0, right=0.0)

    try:
        pg(q, iq)
    except Exception as exc:
        print(f"PDFGetter failed for {file_path.name}: {exc}")
        continue

    for otype in OUTPUTTYPES:
        if otype not in OUTPUT_INFO:
            print(f"  Unknown output type '{otype}', skipping.")
            continue
        ext, header = OUTPUT_INFO[otype]
        data = getattr(pg, otype)
        out_path = SAVE_PATH / file_path.with_suffix(ext).name
        np.savetxt(out_path, np.column_stack(data), header=header, comments="# ")

    print(f"Saved: {file_path.stem}")

print("Done.")
