# Biomarker Relationships & Reference Ranges

This project provides a complete pipeline for working with multi-omics data to derive healthy reference models for biomarker relationships. It includes modules for:

1. **Data Preprocessing**  
   Merge multiple omics datasets (e.g., proteomics, metabolomics), apply winsorization to limit outlier effects, and split the data into training and test sets.

2. **Model Training**  
   Train XGBoost regression models for each feature (biomarker), with hyperparameters that can be tuned directly via the command line.

3. **Bootstrap Analysis**  
   Generate bootstrapped reference ranges for residuals from the prediction models.

All modules are designed to work together in a modular fashion.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training with Hyperparameter Tuning](#model-training-with-hyperparameter-tuning)
  - [Bootstrap Analysis](#bootstrap-analysis)
- [Repository Structure](#repository-structure)


---

## Installation

### Prerequisites

- **Python 3.6+**
- **Git** (for cloning the repository)

### Steps

1. **Clone the repository:**

```bash
git clone https://github.com/<your-username>/biomarker_relationships_reference_ranges.git
```

2. **Navigate to the repository directory:**

```bash
cd biomarker_relationships_reference_ranges
```

3. **Create and activate a virtual environment (recommended):**

- On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

- On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

4. **Install dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Your `requirements.txt` should include (at minimum):

```
numpy
pandas
scikit-learn
xgboost
joblib
scipy
```

---

## Usage

### Data Preprocessing

The `data_preprocessing.py` module merges multiple omics datasets, applies winsorization, and splits the data into training and test sets.

**Example Command:**
```bash
python data_preprocessing.py --omics data/proteomics_df_scaled_imputed.tsv data/metabolomics_df_scaled_imputed.tsv
```
- `--omics`: Space-separated list of file paths to your omics datasets.  
- The script merges them on the default key (`public_client_id`) and creates `train_df.csv` and `test_df.csv` in the `output/` folder.

---

### Model Training with Hyperparameter Tuning

The `model_training.py` module loads the preprocessed training data (`train_df.csv`) from the `output` folder and trains an XGBoost model for each feature. Hyperparameters can be tuned via command-line arguments.

**Example Command:**
```bash
python model_training.py --n_estimators 500 --learning_rate 0.005 --max_depth 6
```
Any hyperparameter not specified will default to the values in the script. Trained models are saved in `output/xgb_models`, and a summary of performance is written to `xgb_results.csv`.

---

### Bootstrap Analysis

*(Optional – if implemented in your workflow.)*

The `bootstrap_analysis.py` module calculates bootstrapped reference ranges for residuals from the trained models.

**Example Command:**
```bash
python bootstrap_analysis.py --n_bootstraps 500 --lower_bootstrap_percentiles 5 50 95 --upper_bootstrap_percentiles 5 50 95 --iqr_multiplier 1.75
```
- `--n_bootstraps`: Number of bootstrap resamples.  
- `--lower_bootstrap_percentiles` / `--upper_bootstrap_percentiles`: Which percentiles to compute for lower and upper thresholds.  
- `--iqr_multiplier`: Adjust how wide the IQR-based thresholds are.  
- The script outputs a `bootstrap_thresholds.csv` file in the `output` folder.

---

## Repository Structure

```
biomarker_relationships_reference_ranges/
├── biomarker_relationships_reference_ranges/
│   ├── __init__.py               # (if packaging as a module)
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── bootstrap_analysis.py     # (optional)
│   └── data/                     # Your input data files
├── output/                       # Generated outputs (CSV files, models, etc.)
├── requirements.txt
├── README.md
└── LICENSE
```

- **data_preprocessing.py**: Script to merge data, winsorize, and split into train/test.  
- **model_training.py**: Script to train XGBoost models with optional hyperparameter tuning.  
- **bootstrap_analysis.py**: Script to derive bootstrapped reference ranges for residuals.  

---

## License

This project is licensed under the [MIT License](LICENSE). Feel free to modify as needed.

---

## Contact

For questions or suggestions, please contact [Your Name / Email] or create an issue on this GitHub repository.

---

**Enjoy using the Biomarker Relationships & Reference Ranges pipeline!** If you find it helpful, please consider starring the repo or contributing back.
