# Bearing Fault Classification Using LSTM on CWRU Vibration Data

## Overview
This artefact accompanies the conference paper report for Applied Artificial Intelligence (UFMF31-30-3) Task 2. It contains all code, data, and generated figures needed to reproduce the results presented in the paper.

## Environment Requirements
- Python 3.10+ (tested on 3.14.3)
- Required packages:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `scikit-learn`
  - `torch` 
  - `pandas`
  - `psutil`

Install with:
```
pip install numpy scipy matplotlib scikit-learn torch pandas psutil
```

## Dataset
The raw data comes from the Case Western Reserve University (CWRU) Bearing Data Center, obtained via Kaggle.

- **Kaggle link:** [https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets?select=CWRU_48k_load_1_CNN_data.npz]
- 10 `.mat` files in the `raw/` folder
- Drive-end accelerometer signals, 12 kHz sampling rate
- 4 classes: Normal, Ball Fault, Inner Race Fault, Outer Race Fault

## How to Run
1. Make sure the `raw/` folder contains all 10 `.mat` files
2. Open `CWRU_Bearing_Fault_Classification.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells top-to-bottom (Cell - Run All)
4. All figures are saved automatically to the `Figures` folder
5. Printed outputs include classification reports, accuracy metrics, and ablation results

## Project Structure
```
Artefact/
├── README.md                                    # This file
├── CWRU_Bearing_Fault_Classification.ipynb      # Main notebook (run top-to-bottom)
├── raw/                                         # Raw .mat data files (10 files)
│   ├── Time_Normal_1_098.mat                    #   Normal bearing baseline
│   ├── B007_1_123.mat                           #   Ball fault (0.007")
│   ├── B014_1_190.mat                           #   Ball fault (0.014")
│   ├── B021_1_227.mat                           #   Ball fault (0.021")
│   ├── IR007_1_110.mat                          #   Inner Race fault (0.007")
│   ├── IR014_1_175.mat                          #   Inner Race fault (0.014")
│   ├── IR021_1_214.mat                          #   Inner Race fault (0.021")
│   ├── OR007_6_1_136.mat                        #   Outer Race fault (0.007")
│   ├── OR014_6_1_202.mat                        #   Outer Race fault (0.014")
│   └── OR021_6_1_239.mat                        #   Outer Race fault (0.021")
└── Figures/                                     # Generated figures (13 PNGs)
```

## Notebook Pipeline
The notebook is structured into 10 sections:

1. **Environment & Hardware Info** - logs system specs and package versions
2. **CPU Setup, Imports & Seeds** - forces CPU execution, sets random seeds for reproducibility
3. **Data Loading** - loads all 10 `.mat` files, extracts drive-end accelerometer signals
4. **Data Cleaning & Preprocessing** - NaN/Inf checks, outlier detection, windowing, class balance check
5. **Exploratory Data Analysis** - time-domain plots, FFT spectra, spectrograms, amplitude distributions, box plots
6. **Feature Engineering** - extracts 9 hand-crafted features per window (RMS, peak, crest factor, kurtosis, skewness, std dev, dominant freq, mean freq, spectral energy)
7. **Train/Val/Test Split** -70/15/15 stratified split, normalisation from training stats only
8. **Baseline Model (SVM)** - RBF kernel SVM on hand-crafted features
9. **Main Model (LSTM)** - stacked LSTM on raw windowed signals
10. **Ablation & Sensitivity Analysis** - feature removal, window size, and SVM C-parameter experiments

## Outputs
Running the notebook produces:
- 13 figures saved to `Figures`
- Printed classification reports for both SVM and LSTM
- Ablation experiment results (feature removal, window size, C-parameter)
- Model comparison table

## Reproducibility
All random seeds are fixed (NumPy, PyTorch, Python) to ensure identical results across runs. The notebook is designed to run on CPU without requiring a GPU.
