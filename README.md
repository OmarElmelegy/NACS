# ECG Signal Classification using SVM in MATLAB

This repository implements a full pipeline for ECG signal analysis and classification using Support Vector Machines (SVM). The pipeline covers:

- **Dataset Construction:** Extracting 10-second ECG segments for three classes:
  - **NSR** (Normal Sinus Rhythm)
  - **CHF** (Congestive Heart Failure)
  - **ARR** (Arrhythmia)
- **Preprocessing & Visualization:** Filtering the raw signals to remove baseline wander and visualizing both unfiltered and filtered signals.
- **Feature Extraction:** Extracting time and frequency domain features (e.g., Hjorth parameters, LF/HF ratio, PSD entropy, and negative peak count) from each segment.
- **Classification:** Training binary SVM classifiers (NSR vs. CHF and NSR vs. ARR) using an RBF kernel with hyperparameter optimization and 5-fold cross-validation.

> **Note:** The current code uses the first four features (Hjorth Mobility, Hjorth Complexity, LF/HF Ratio, and PSD Entropy) for classification. You can experiment with different feature combinations and even enrich the feature set if needed.

## Table of Contents

- [ECG Signal Classification using SVM in MATLAB](#ecg-signal-classification-using-svm-in-matlab)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Repository Structure](#repository-structure)
  - [Requirements](#requirements)
  - [Usage](#usage)
    - [Step 1: Dataset Construction](#step-1-dataset-construction)
    - [Step 2: Preprocessing and Visualization](#step-2-preprocessing-and-visualization)
    - [Step 3: Feature Extraction](#step-3-feature-extraction)
    - [Step 4: SVM Classification with Hyperparameter Optimization](#step-4-svm-classification-with-hyperparameter-optimization)
  - [Performance Metrics](#performance-metrics)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## Overview

This project processes an ECG dataset (`ECGData.mat`) to extract 10-second segments for three classes: NSR, CHF, and ARR. The pipeline applies a 6th-order Butterworth high-pass filter to remove baseline wander, extracts several features from the filtered signals, and then uses these features to train SVM classifiers.

The classifiers are built using an RBF kernel, and hyperparameters (BoxConstraint and KernelScale) are optimized using MATLAB’s built-in optimization routines. The model is evaluated using 5-fold cross-validation, and performance metrics such as Accuracy, True Positive Rate (TPR), False Positive Rate (FPR), and Precision are displayed.

## Repository Structure

```
├── README.md                     # This file
├── NACS.m                        # Main MATLAB script implementing the pipeline
├── ECGData.mat                   # Input ECG dataset (must be placed in the working directory)
└── Selected_ECG_Segments.mat     # Generated dataset of ECG segments (saved by the script)
```

## Requirements

- MATLAB (R2018a or later is recommended)
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox
- An ECG dataset file named `ECGData.mat` in your working directory

## Usage

1. **Prepare the Dataset:**  
   Ensure `ECGData.mat` is in your MATLAB working directory.

2. **Run the Script:**  
   Open and run `NACS.m` in MATLAB. The script is divided into four main sections:

### Step 1: Dataset Construction

- **What It Does:**  
  Loads `ECGData.mat` and extracts 10-second segments (at 128 Hz) for NSR, CHF, and ARR classes.  
- **Details:**  
  - Segments are randomly extracted from up to 30 subjects per class.
  - A total of 2000 segments per class are collected.
  - The extracted segments are saved to `Selected_ECG_Segments.mat`.

### Step 2: Preprocessing and Visualization

- **What It Does:**  
  Applies a 6th-order Butterworth high-pass filter (cutoff frequency 0.5 Hz) using zero-phase filtering (`filtfilt`) to remove baseline wander.
- **Visualization:**  
  The script plots sample ECG signals before and after filtering for visual inspection.

### Step 3: Feature Extraction

- **What It Does:**  
  For each filtered segment, the following features are extracted:
  - **Hjorth Mobility** and **Hjorth Complexity**
  - **LF/HF Ratio** (ratio of power in low-frequency vs. high-frequency bands)
  - **PSD Entropy** (entropy of the power spectral density)
  - **Negative Peak Count** (count of negative peaks below a threshold)
- **Note:**  
  For classification, the first four features are used (you can experiment with including additional features).

### Step 4: SVM Classification with Hyperparameter Optimization

- **What It Does:**  
  The extracted features are normalized (z-score normalization) and then used to train two binary classifiers:
  - **NSR vs. CHF**
  - **NSR vs. ARR**
- **Hyperparameter Optimization:**  
  MATLAB’s hyperparameter optimization is used to tune the `BoxConstraint` and `KernelScale` for an RBF kernel SVM.
- **Evaluation:**  
  The classifiers are evaluated using 5-fold cross-validation. Performance metrics (Accuracy, TPR, FPR, Precision) are displayed in a table.

## Performance Metrics

After running the script, you may see output similar to:

```
Optimized NSR vs CHF parameters: BoxConstraint = 625.8608, KernelScale = 1.2307
Optimized NSR vs ARR parameters: BoxConstraint = 793.8773, KernelScale = 1.8598
      Comparison      Accuracy (%)    TPR (%)    FPR (%)    Precision (%)
    ______________    ____________    _______    _______    _____________

    {'NSR vs CHF'}        84.6        51.475       8.5         59.767    
    {'NSR vs ARR'}         100            60         0             60    
```

These metrics may vary based on the dataset and segmentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project was inspired by research in ECG signal processing and classification.
- MATLAB documentation and examples for signal processing and machine learning were invaluable in developing this pipeline.

---

Feel free to modify or extend the code (e.g., by experimenting with additional features or alternate classifiers) to further optimize performance. If you have any questions or suggestions, please open an issue in the repository.

---