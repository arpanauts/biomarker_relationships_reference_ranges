import os
import argparse
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split

def load_omics_data(file_paths, merge_key='public_client_id', sep='\t', comment_char='#'):
    """
    Load and merge multiple omics datasets on a shared key.

    Parameters:
        file_paths (list of str): List of file paths to omics datasets.
        merge_key (str): The column name on which to merge the datasets.
        sep (str): Delimiter for the data files (e.g., '\t' for TSV).
        comment_char (str): Character to indicate comment lines in the data files.

    Returns:
        DataFrame: A single merged DataFrame containing all datasets joined on merge_key.
    """
    if not file_paths:
        raise ValueError("No file paths provided for omics data.")
    
    # Load the first dataset
    merged_df = pd.read_csv(file_paths[0], sep=sep, comment=comment_char)
    
    # Iteratively merge each subsequent dataset
    for path in file_paths[1:]:
        df = pd.read_csv(path, sep=sep, comment=comment_char)
        merged_df = merged_df.merge(df, on=merge_key, how='inner')
    
    return merged_df

def winsorize_df_with_stats(df, lower_percentile=0.01, upper_percentile=0.99, exclude_cols=None):
    """
    Winsorize numeric columns of a DataFrame and record statistics about the capping.
    
    Parameters:
        df (DataFrame): The input DataFrame.
        lower_percentile (float): Lower percentile cutoff.
        upper_percentile (float): Upper percentile cutoff.
        exclude_cols (list, optional): List of columns to exclude from winsorization.
        
    Returns:
        tuple: (winsorized_df, capping_stats_df)
            winsorized_df (DataFrame): DataFrame with winsorized columns.
            capping_stats_df (DataFrame): DataFrame with capping statistics per feature.
    """
    exclude_cols = exclude_cols or []
    winsorized_df = df.copy()
    capping_stats = []

    for column in winsorized_df.columns:
        # Only winsorize numeric columns that are not excluded
        if column not in exclude_cols and winsorized_df[column].dtype in ['float64', 'int64']:
            original_values = winsorized_df[column]
            lower_limit = original_values.quantile(lower_percentile)
            upper_limit = original_values.quantile(upper_percentile)

            lower_capped_count = (original_values < lower_limit).sum()
            upper_capped_count = (original_values > upper_limit).sum()
            total_capped = lower_capped_count + upper_capped_count

            # Apply winsorization (limits are specified as proportions)
            winsorized_df[column] = winsorize(original_values, limits=(lower_percentile, 1 - upper_percentile))

            capping_stats.append({
                'Feature': column,
                'Lower_Capped_Count': lower_capped_count,
                'Upper_Capped_Count': upper_capped_count,
                'Total_Capped': total_capped
            })

    capping_stats_df = pd.DataFrame(capping_stats)
    return winsorized_df, capping_stats_df

def preprocess_data(merged_df, id_column='public_client_id', test_size=0.2,
                    random_state=15, variance_threshold=0.01, correlation_threshold=0.9,
                    lower_percentile=0.01, upper_percentile=0.99):
    """
    Process merged data by winsorizing, splitting into training/testing sets,
    and removing low variance and highly correlated features.
    
    Parameters:
        merged_df (DataFrame): Merged dataset containing all omics data.
        id_column (str): Identifier column to exclude from processing.
        test_size (float): Fraction of the data to use as test set.
        random_state (int): Random state for reproducibility.
        variance_threshold (float): Threshold for removing low variance features.
        correlation_threshold (float): Threshold for dropping highly correlated features.
        lower_percentile (float): Winsorization lower percentile cutoff.
        upper_percentile (float): Winsorization upper percentile cutoff.
    
    Returns:
        tuple: (train_df, test_df, low_variance_features, high_correlation_features, capping_stats_df)
    """
    # Winsorize all columns except the identifier using specified percentiles
    winsorized_df, capping_stats_df = winsorize_df_with_stats(
        merged_df, lower_percentile=lower_percentile, upper_percentile=upper_percentile, exclude_cols=[id_column]
    )

    # Separate features from the identifier
    features = winsorized_df.drop(columns=[id_column])
    ids = winsorized_df[id_column]

    # Split the data into training and test sets
    features_train, features_test, ids_train, ids_test = train_test_split(
        features, ids, test_size=test_size, random_state=random_state
    )

    # Remove low variance features
    low_variance_features = [col for col in features_train if features_train[col].std() < variance_threshold]
    features_train = features_train.drop(columns=low_variance_features)
    features_test = features_test.drop(columns=low_variance_features, errors='ignore')

    # Identify highly correlated features (upper triangle of correlation matrix)
    correlation_matrix = features_train.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    high_correlation_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]

    # Drop highly correlated features
    features_train = features_train.drop(columns=high_correlation_features)
    features_test = features_test.drop(columns=high_correlation_features, errors='ignore')

    # Reattach the identifier
    train_df = features_train.copy()
    train_df[id_column] = ids_train.values
    test_df = features_test.copy()
    test_df[id_column] = ids_test.values

    return train_df, test_df, low_variance_features, high_correlation_features, capping_stats_df

def main():
    """
    Main function to merge and preprocess omics datasets.
    This script accepts command-line arguments to adjust winsorization percentiles and train/test split proportion.
    """
    parser = argparse.ArgumentParser(
        description="Merge and preprocess omics datasets with adjustable winsorization and train/test split parameters."
    )
    parser.add_argument("--omics", nargs='+', required=True, help="List of file paths for omics datasets")
    parser.add_argument("--id_column", type=str, default="public_client_id", help="Identifier column name")
    parser.add_argument("--lower_percentile", type=float, default=0.01, help="Winsorization lower percentile cutoff")
    parser.add_argument("--upper_percentile", type=float, default=0.99, help="Winsorization upper percentile cutoff")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data to use as test set")
    parser.add_argument("--random_state", type=int, default=15, help="Random state for train/test split")
    parser.add_argument("--variance_threshold", type=float, default=0.01, help="Threshold for low variance features")
    parser.add_argument("--correlation_threshold", type=float, default=0.9, help="Threshold for dropping highly correlated features")
    args = parser.parse_args()
    
    # Load and merge the provided omics datasets
    merged_df = load_omics_data(args.omics, merge_key=args.id_column)
    
    # Preprocess the merged data using the command-line parameters
    train_df, test_df, low_var_feats, high_corr_feats, capping_stats_df = preprocess_data(
        merged_df,
        id_column=args.id_column,
        test_size=args.test_size,
        random_state=args.random_state,
        variance_threshold=args.variance_threshold,
        correlation_threshold=args.correlation_threshold,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile
    )
    
    # Create the output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Save processed data and statistics
    train_df.to_csv(os.path.join('output', 'train_df.csv'), index=False)
    test_df.to_csv(os.path.join('output', 'test_df.csv'), index=False)
    capping_stats_df.to_csv(os.path.join('output', 'capping_stats.csv'), index=False)
    
    # Print summary information
    print(f"Training set size: {train_df.shape[0]} rows")
    print(f"Test set size: {test_df.shape[0]} rows")
    print(f"Low variance features dropped: {low_var_feats}")
    print(f"Highly correlated features dropped: {high_corr_feats}")

if __name__ == '__main__':
    main()
