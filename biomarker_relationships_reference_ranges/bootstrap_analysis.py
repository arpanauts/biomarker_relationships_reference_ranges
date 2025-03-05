import os
import argparse
import numpy as np
import pandas as pd
from joblib import load
from scipy.stats.mstats import winsorize

def derive_bootstrap_reference_ranges(test_df, models_dir, n_bootstraps=1000, 
                                        lower_bootstrap_percentiles=(5, 50, 95), 
                                        upper_bootstrap_percentiles=(5, 50, 95), 
                                        iqr_multiplier=1.5, id_column='public_client_id',
                                        winsorize_limits=(0, 0)):
    """
    Derive bootstrap-based reference ranges for residuals of test data using pretrained models.
    
    Parameters:
        test_df (DataFrame): Test dataset containing features and an identifier column.
        models_dir (str): Directory where pretrained models are stored.
        n_bootstraps (int): Number of bootstrap resamples.
        lower_bootstrap_percentiles (tuple or list): Percentiles to compute for the lower bound from the bootstrap distribution.
        upper_bootstrap_percentiles (tuple or list): Percentiles to compute for the upper bound from the bootstrap distribution.
        iqr_multiplier (float): Multiplier to compute IQR-based thresholds.
        id_column (str): Name of the identifier column to skip.
        winsorize_limits (tuple): Limits for winsorizing residuals (e.g., (0, 0) disables winsorization).
        
    Returns:
        DataFrame: A summary DataFrame containing baseline IQR thresholds, 
                   bootstrapped percentile values, and the final chosen cutoffs for each feature.
    """
    bootstrap_results = []
    
    for feature in test_df.columns:
        # Skip identifier column
        if feature == id_column:
            continue
        
        # Load trained model for the feature
        model_filename = os.path.join(models_dir, f"xgb_model_{feature}.joblib")
        if not os.path.exists(model_filename):
            print(f"Warning: Model file for {feature} not found. Skipping...")
            continue
        model = load(model_filename)
        
        # Prepare test data: drop id and target feature to form predictors
        X_test = test_df.drop(columns=[id_column, feature], errors="ignore")
        predictions = model.predict(X_test)
        
        # Observed values and residuals for the target feature
        observed = test_df[feature]
        residuals = observed - predictions
        
        # Optionally winsorize extreme residuals (disable by setting limits to (0,0))
        residuals_winsorized = winsorize(residuals, limits=winsorize_limits)
        
        # Compute raw IQR-based thresholds (baseline)
        q1, q3 = np.percentile(residuals_winsorized, [25, 75])
        iqr = q3 - q1
        lower_iqr = q1 - iqr_multiplier * iqr
        upper_iqr = q3 + iqr_multiplier * iqr
        
        # Bootstrap testing: resample residuals and compute IQR thresholds for each sample
        bootstrapped_thresholds = []
        for _ in range(n_bootstraps):
            sample = np.random.choice(residuals_winsorized, size=len(residuals_winsorized), replace=True)
            q1_sample, q3_sample = np.percentile(sample, [25, 75])
            iqr_sample = q3_sample - q1_sample
            lower_sample = q1_sample - iqr_multiplier * iqr_sample
            upper_sample = q3_sample + iqr_multiplier * iqr_sample
            bootstrapped_thresholds.append([lower_sample, upper_sample])
        
        bootstrapped_thresholds = np.array(bootstrapped_thresholds)
        lower_all = bootstrapped_thresholds[:, 0]
        upper_all = bootstrapped_thresholds[:, 1]
        
        # Compute the bootstrap percentiles for lower and upper thresholds separately
        lower_stats = np.percentile(lower_all, lower_bootstrap_percentiles)
        upper_stats = np.percentile(upper_all, upper_bootstrap_percentiles)
        
        # Use the median (second value) as the final threshold if available,
        # otherwise default to the first value in case only one percentile is provided.
        final_lower = lower_stats[1] if len(lower_stats) >= 2 else lower_stats[0]
        final_upper = upper_stats[1] if len(upper_stats) >= 2 else upper_stats[0]
        
        # Create a dictionary of results for the current feature
        feature_results = {
            "Feature": feature,
            "Lower_IQR": lower_iqr,
            "Upper_IQR": upper_iqr,
            "Final_Lower_Threshold": final_lower,
            "Final_Upper_Threshold": final_upper,
        }
        # Add bootstrapped lower statistics
        for perc, val in zip(lower_bootstrap_percentiles, lower_stats):
            feature_results[f"Lower_Boot_{perc}th"] = val
        # Add bootstrapped upper statistics
        for perc, val in zip(upper_bootstrap_percentiles, upper_stats):
            feature_results[f"Upper_Boot_{perc}th"] = val
        
        bootstrap_results.append(feature_results)
    
    results_df = pd.DataFrame(bootstrap_results).set_index("Feature")
    return results_df

def main():
    """
    Load the test data, derive bootstrap reference ranges for residuals using pretrained models,
    and save the results to an output CSV.
    
    This script accepts command-line arguments to adjust the bootstrapping parameters.
    """
    parser = argparse.ArgumentParser(
        description="Derive bootstrap reference ranges for residuals using pretrained models."
    )
    parser.add_argument("--n_bootstraps", type=int, default=1000,
                        help="Number of bootstrap resamples (default: 1000)")
    parser.add_argument("--lower_bootstrap_percentiles", nargs='+', type=float, default=[5, 50, 95],
                        help="Percentiles for the lower bound (e.g., 5 50 95)")
    parser.add_argument("--upper_bootstrap_percentiles", nargs='+', type=float, default=[5, 50, 95],
                        help="Percentiles for the upper bound (e.g., 5 50 95)")
    parser.add_argument("--iqr_multiplier", type=float, default=1.5,
                        help="IQR multiplier for threshold calculation (default: 1.5)")
    parser.add_argument("--winsorize_limits", nargs=2, type=float, default=[0, 0],
                        help="Winsorize limits as two numbers (default: 0 0 to disable winsorization)")
    parser.add_argument("--id_column", type=str, default="public_client_id",
                        help="Identifier column name (default: public_client_id)")
    args = parser.parse_args()
    
    # Convert lists to tuples where needed
    lower_bootstrap_percentiles = tuple(args.lower_bootstrap_percentiles)
    upper_bootstrap_percentiles = tuple(args.upper_bootstrap_percentiles)
    winsorize_limits = tuple(args.winsorize_limits)
    
    # Define relative paths for test data and model storage
    test_data_path = os.path.join("output", "test_df.csv")
    models_dir = os.path.join("output", "xgb_models")
    
    # Load test data
    test_df = pd.read_csv(test_data_path)
    
    # Derive the bootstrap reference ranges with command-line adjustable parameters
    bootstrap_thresholds_df = derive_bootstrap_reference_ranges(
        test_df, 
        models_dir, 
        n_bootstraps=args.n_bootstraps, 
        lower_bootstrap_percentiles=lower_bootstrap_percentiles, 
        upper_bootstrap_percentiles=upper_bootstrap_percentiles,
        iqr_multiplier=args.iqr_multiplier, 
        id_column=args.id_column,
        winsorize_limits=winsorize_limits
    )
    
    # Save the results to a CSV file
    output_file = os.path.join("output", "bootstrap_thresholds.csv")
    bootstrap_thresholds_df.to_csv(output_file)
    
    print("Bootstrap reference ranges have been computed and saved:")
    print(bootstrap_thresholds_df)

if __name__ == '__main__':
    main()
