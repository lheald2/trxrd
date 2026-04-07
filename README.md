# TRXRD

**TRXRD** is a Python package for processing and analyzing time-resolved X-ray scattering (TR-XRD) and total scattering datasets, especially those collected at the 7ID station at the Advanced Photon Source. It provides tools for batch image processing, masking, normalization, background subtraction, and azimuthal integration, enabling efficient extraction of time-dependent structural information.

---

## Features

* Batch processing of detector images
* Beam stop and detector masking
* X-ray removal and outlier filtering
* Automatic center detection
* Azimuthal integration (via pyFAI or custom methods)
* Background subtraction with normalization
* Time-resolved data handling and averaging
* *G(r)* and *dG(r)* generation from reduced structure factors
* Export to `.dat` and `.h5` formats

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
pip install numpy scipy matplotlib h5py pyFAI tifffile
```

---

## Quick Start
The main package uses a globals.py file which are incorporated into the interative notebooks and the batch processing python files. Make sure your globals are set before running batch processing algorithms.

For fast analysis of time-resolved x-ray diffraction data sets, use the `Fast_Analysis.ipynb` notebook which imports the data, removes any outlier images, then averages the data by delay for faster processing. 

To visualize the data processing steps and interactively adjust parameters, use the `Data_Analysis.ipynb` notebook which applies the standard processing steps to all images before grouping by delay. This notebook will help you set all your global variables. Once you're satisfied with the globals, you can then move on to batch processing your data using the python files like `process_data.py` and `batch_process_dat_files.py`. 

`process_data.py` runs through the full processing algorithm in the `Data_Analysis.ipynb` file without visualization and saves the results to an h5 file. The results can then be visualized with the `Analyze_Processed_Data.ipynb` file. 

The `batch_process_dat_files.py` program reads in the images, applies masks and other cleanup steps, then saves the normalized azimuthal averages to individual .dat files for using with other programs like *PDFgetX3*. 

---

## Command Line Usage

You can run the batch processing script directly from the command line:

```bash
python batch_process_dat_files.py
```

This will execute the default processing pipeline defined in the script (e.g., loading data, applying masks, performing azimuthal integration, and saving outputs).

---

## Workflow Overview

1. **Load images**
2. **Apply masks** (beam stop, detector defects, removed x-rays)
3. **Perform azimuthal integration**
4. **Normalize intensities**
5. **Subtract background**
6. **Group and average by time delay**
7. **Save processed data**

---

## Core Functionality

### `globals.py`

Configuration type file where global variables are defined. Each other notebook and python file relies on these global variables. 

### `Fast_Analysis.ipynb`

Interactive notebook for quickly processing time-resolved x-ray scattering datasets. Good for quickly visualizing results. 

### `Find_Centers.ipynb`

Interactive notebook for finding precise centers from a data set. This can be a time consuming process so I've broken it into its own module to be done once then the found centers can be set in the `globals.py` file for future use. 

### `Data_Analysis.ipynb`

Interactive notebook for processing .tif files from a time-resolved x-ray diffraction experiment. 

### `batch_process_dat_files.py`

Program for importing .tif image files and outputting azimuthally averaged and normalized .dat files. 

### `process_data.py`

Workhorse program which preforms all the data processing steps in the `Data_Analysis.ipynb` file without visualization. Saves the results in a .h5 file for further analysis and visualization. Only do this once all your global variables are set and you are happy with the processing steps. 

### `Analyze_Processed_Data.ipynb`

Interactive notebook for visualizing and analyzing the output of the `process_data.py` program. 


---

## Performance Tips

* Use precomputed combined masks to avoid recomputation
* Parallelize batch processing with `concurrent.futures`
* Avoid saving unnecessary intermediate data
* Use HDF5 for large datasets

---

## Dependencies

* numpy
* scipy
* matplotlib
* h5py
* pyFAI
* tifffile

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

For questions or collaborations, please contact:

**Your Name**
Email: [lauren.f.heald@gmail.com](mailto:lauren.f.heald@gmail.com)

---

## Acknowledgments

Dr. Burak Guzelturk - APS Physicist 
