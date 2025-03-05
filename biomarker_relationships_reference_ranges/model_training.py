import os
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed, dump
import argparse

# Default hyperparameters for the XGBoost models.
DEFAULT_XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.01,
    "max_depth": 4,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 1.5,
    "gamma": 0.5,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}

def train_xgb_models(train_df, id_column='public_client_id', models_output_dir=None, xgb_params=None):
    """
    Train an XGBoost regression model for each feature in the training data.
    Each model predicts the target feature using all other features as predictors.
    
    Parameters:
        train_df (DataFrame): Training dataset that includes an identifier column.
        id_column (str): Column name that uniquely identifies each observation.
        models_output_dir (str, optional): Directory where trained models will be saved.
            Defaults to 'output/xgb_models'.
        xgb_params (dict, optional): Dictionary of hyperparameters for the XGBoost model.
            Defaults to DEFAULT_XGB_PARAMS.
    
    Returns:
        DataFrame: A summary DataFrame containing each feature's model, training MSE,
                   and cross-validated R² score.
    """
    # Suppress specific warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Set up the output directory for saving models
    if models_output_dir is None:
        models_output_dir = os.path.join("output", "xgb_models")
    os.makedirs(models_output_dir, exist_ok=True)
    
    # Use provided hyperparameters or the default ones
    if xgb_params is None:
        xgb_params = DEFAULT_XGB_PARAMS

    # Prepare training data by dropping the identifier column
    X_full = train_df.drop(columns=[id_column], errors='ignore')
    
    # (Optional) Drop low-variance features if any remain
    low_variance_features = [col for col in X_full if X_full[col].std() < 0.01]
    if low_variance_features:
        X_full = X_full.drop(columns=low_variance_features)
    
    def train_xgb_for_feature(target_feature):
        """
        Train an XGBoost model for a single target feature.
        
        Parameters:
            target_feature (str): The feature to be predicted.
            
        Returns:
            tuple: (target_feature, trained model, training MSE, mean CV R² score)
        """
        # Define predictors (all columns except the target) and target variable
        X = X_full.drop(columns=[target_feature])
        y = X_full[target_feature]
        
        # Initialize the XGBoost regressor with the provided hyperparameters
        model = XGBRegressor(**xgb_params)
        
        # Evaluate the model using 5-fold cross-validation (R² score)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
        cv_mean = np.mean(cv_scores)
        
        # Fit the model on the full dataset
        model.fit(X, y)
        
        # Compute training Mean Squared Error
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        
        # Save the trained model to a file
        model_filename = os.path.join(models_output_dir, f"xgb_model_{target_feature}.joblib")
        dump(model, model_filename)
        
        return target_feature, model, mse, cv_mean

    # Train a model for each feature in parallel
    results = Parallel(n_jobs=-1)(
        delayed(train_xgb_for_feature)(feature) for feature in X_full.columns
    )
    
    # Convert the results into a summary DataFrame
    results_df = pd.DataFrame(results, columns=['Feature', 'Model', 'MSE', 'CV_R2'])
    return results_df

def main():
    """
    Load the preprocessed training data, train models for each feature,
    and save both the models and a summary of performance metrics.
    Enables hyperparameter tuning via command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train XGBoost models for each feature using preprocessed training data."
    )
    parser.add_argument("--n_estimators", type=int, help="Number of boosting rounds", default=None)
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=None)
    parser.add_argument("--max_depth", type=int, help="Maximum tree depth", default=None)
    parser.add_argument("--min_child_weight", type=int, help="Minimum sum of instance weight", default=None)
    parser.add_argument("--subsample", type=float, help="Subsample ratio", default=None)
    parser.add_argument("--colsample_bytree", type=float, help="Column sample ratio", default=None)
    parser.add_argument("--reg_alpha", type=float, help="L1 regularization coefficient", default=None)
    parser.add_argument("--reg_lambda", type=float, help="L2 regularization coefficient", default=None)
    parser.add_argument("--gamma", type=float, help="Minimum loss reduction to split further", default=None)
    
    args = parser.parse_args()
    
    # Build hyperparameters dictionary
    xgb_params = DEFAULT_XGB_PARAMS.copy()
    if args.n_estimators is not None:
        xgb_params['n_estimators'] = args.n_estimators
    if args.learning_rate is not None:
        xgb_params['learning_rate'] = args.learning_rate
    if args.max_depth is not None:
        xgb_params['max_depth'] = args.max_depth
    if args.min_child_weight is not None:
        xgb_params['min_child_weight'] = args.min_child_weight
    if args.subsample is not None:
        xgb_params['subsample'] = args.subsample
    if args.colsample_bytree is not None:
        xgb_params['colsample_bytree'] = args.colsample_bytree
    if args.reg_alpha is not None:
        xgb_params['reg_alpha'] = args.reg_alpha
    if args.reg_lambda is not None:
        xgb_params['reg_lambda'] = args.reg_lambda
    if args.gamma is not None:
        xgb_params['gamma'] = args.gamma
    
    # Load training data from the output directory (saved by the preprocessing module)
    train_data_path = os.path.join("output", "train_df.csv")
    train_df = pd.read_csv(train_data_path)
    
    # Train XGBoost models using the provided (or default) hyperparameters
    results_df = train_xgb_models(train_df, xgb_params=xgb_params)
    
    # Save the results summary as CSV for further tracking
    results_output_path = os.path.join("output", "xgb_results.csv")
    results_df.to_csv(results_output_path, index=False)
    
    # Print a summary of results
    print("XGBoost Model Training Completed")
    print(results_df[['Feature', 'MSE', 'CV_R2']])

if __name__ == '__main__':
    main()
