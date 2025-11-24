import pandas as pd
import numpy as np

def calculate_metrics(errors, threshold=2.0):
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    mse = np.mean(errors**2)
    accuracy = np.mean(errors < threshold)
    max_error = np.max(errors)
    std_error = np.std(errors)   # Standardabweichung hinzufügen
    return mean_error, median_error, mse, accuracy, max_error, std_error

def compute_overall_metrics(df):
    # Create an empty DataFrame to hold the metrics
    metrics_df = pd.DataFrame(
        columns=["Mean Error", "Median Error", "MSE", "Accuracy", "Max Error", "Std Error"]
    )

    # Calculate metrics for each error type
    for error_type in [
        "coarse_error",
        "refined_error",
        "coarse_proj_error",
        "refined_proj_error",
    ]:
        metrics_df.loc[error_type] = calculate_metrics(df[error_type].values)

    return metrics_df


# Beispiel: CSV laden und Metriken berechnen
if __name__ == "__main__":
    path = "neighbor/evaluation/neighbor-subreg-project_gt-no_freeze-standard_architecture-excel_outliers_exclude/test/version2_epoch124/results.csv"
    df = pd.read_csv(path)
    exclude_pois = {40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50}

    df_filtered = df[~df["poi_idx"].isin(exclude_pois)]
    metrics = compute_overall_metrics(df_filtered)
    save_path = "neighbor/evaluation/neighbor-subreg-project_gt-no_freeze-standard_architecture-excel_outliers_exclude/test/version2_epoch124/overall_metrics_neighbors_test.csv"
    metrics.to_csv(save_path)
    print(metrics)
