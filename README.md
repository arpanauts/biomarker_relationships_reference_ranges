# Biomarker Relationships & Reference Ranges

This project provides a complete pipeline for working with multi-omics data to derive healthy reference models for biomarker relationships. It includes modules for:

1. **Data Preprocessing**  
   Merge multiple omics datasets (e.g., proteomics, metabolomics), apply winsorization to limit outlier effects, and split the data into training and test sets.

2. **Model Training**  
   Train XGBoost regression models for each feature (biomarker), with hyperparameters that can be tuned directly via the command line.

3. **Bootstrap Analysis (Optional)**  
   Generate bootstrapped reference ranges for residuals from the prediction models (if you choose to use this module).

All modules are designed to work together in a modular fashion and use relative paths for portability.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training with Hyperparameter Tuning](#model-training-with-hyperparameter-tuning)
  - [Bootstrap Analysis](#bootstrap-analysis)
- [Repository Structure](#repository-structure)
- [License](#license)
- [Contact](#contact)

---

## Installation

### Prerequisites

- **Python 3.6+**
- **Git** (for cloning the repository)

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/biomarker_relationships_reference_ranges.git
