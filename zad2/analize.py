import dataframe_image as dfi
import os
import pandas as pd

os.makedirs("results_analysis", exist_ok=True)

df = pd.read_csv("results/experiment_results.csv")

df[['data_source', 'preprocessing']] = df['dataset'].str.split('_', expand=True)

anomaly_condition = (
        (df['activation'] == 'tanh') & 
        (df['dataset'] == 'Dane1_standardise')
    )
    
df_anomaly = df[anomaly_condition]
df_filtered = df[~anomaly_condition].copy()

group_cols = ['activation', 'optimizer', 'structure']
df_agg = df_filtered.groupby(group_cols).agg(
        mse_train_mean=('mse_train', 'mean'),
        mse_train_std=('mse_train', 'std'),
        mse_test_mean=('mse_test', 'mean'),
        mse_test_std=('mse_test', 'std'),
        epochs_run_mean=('epochs_run', 'mean'),
        fit_time_sec_mean=('fit_time_sec', 'mean')
    ).reset_index()


exports = [
    ('mse_test_mean', True, '10_best_mean.png'),
    ('mse_test_mean', False, '10_worst_mean.png'),
    ('mse_test_std', True, '10_best_std.png'),
    ('mse_test_std', False, '10_worst_std.png'),
    ('epochs_run_mean', False, '10_most_epoch.png'),
    ('epochs_run_mean', True, '10_least_epoch.png'),
    ('fit_time_sec_mean', False, '10_longest_training.png'),
    ('fit_time_sec_mean', True, '10_shortest_training.png')
]

group_cols = ['activation', 'optimizer', 'structure', 'data_source']
df_agg = df_filtered.groupby(group_cols).agg(
    mse_train_mean=('mse_train', 'mean'),
    mse_train_std=('mse_train', 'std'),
    mse_test_mean=('mse_test', 'mean'),
    mse_test_std=('mse_test', 'std'),
    epochs_run_mean=('epochs_run', 'mean'),
    fit_time_sec_mean=('fit_time_sec', 'mean')
).reset_index()

for col, ascending, filename in exports:
    top_dane1 = df_agg[df_agg['data_source']=='Dane1'].sort_values(by=col, ascending=ascending).head(10)
    top_dane2 = df_agg[df_agg['data_source']=='Dane2'].sort_values(by=col, ascending=ascending).head(10)
    combined = pd.concat([top_dane1, top_dane2], ignore_index=True)
    dfi.export(combined, f"results_analysis/{filename}")

group_cols = ['activation', 'data_source']
df_activation = df_filtered.groupby(group_cols).agg(
        mse_train_mean=('mse_train', 'mean'),
        mse_train_std=('mse_train', 'std'),
        mse_test_mean=('mse_test', 'mean'),
        mse_test_std=('mse_test', 'std'),
        epochs_run_mean=('epochs_run', 'mean'),
        fit_time_sec_mean=('fit_time_sec', 'mean')
    ).reset_index()
dfi.export(df_activation.sort_values(by='mse_test_mean', ascending=True), "results_analysis/activations.png")

group_cols = ['optimizer']
df_optimizer = df_filtered.groupby(group_cols).agg(
        mse_train_mean=('mse_train', 'mean'),
        mse_train_std=('mse_train', 'std'),
        mse_test_mean=('mse_test', 'mean'),
        mse_test_std=('mse_test', 'std'),
        epochs_run_mean=('epochs_run', 'mean'),
        fit_time_sec_mean=('fit_time_sec', 'mean')
    ).reset_index()
dfi.export(df_optimizer.sort_values(by='mse_test_mean', ascending=True), "results_analysis/optimizers.png")

group_cols = ['structure']
df_structure = df_filtered.groupby(group_cols).agg(
        mse_train_mean=('mse_train', 'mean'),
        mse_train_std=('mse_train', 'std'),
        mse_test_mean=('mse_test', 'mean'),
        mse_test_std=('mse_test', 'std'),
        epochs_run_mean=('epochs_run', 'mean'),
        fit_time_sec_mean=('fit_time_sec', 'mean')
    ).reset_index()
dfi.export(df_structure.sort_values(by='mse_test_mean', ascending=True), "results_analysis/structures.png")

group_cols = ['activation', 'data_source', 'preprocessing']
df_activation = df_filtered.groupby(group_cols).agg(
        mse_train_mean=('mse_train', 'mean'),
        mse_train_std=('mse_train', 'std'),
        mse_test_mean=('mse_test', 'mean'),
        mse_test_std=('mse_test', 'std'),
        epochs_run_mean=('epochs_run', 'mean'),
        fit_time_sec_mean=('fit_time_sec', 'mean')
    ).reset_index()
dfi.export(df_activation.sort_values(by='mse_test_mean', ascending=True), "results_analysis/preprocessing_activations.png")

group_cols = ['optimizer', 'preprocessing']
df_optimizer = df_filtered.groupby(group_cols).agg(
        mse_train_mean=('mse_train', 'mean'),
        mse_train_std=('mse_train', 'std'),
        mse_test_mean=('mse_test', 'mean'),
        mse_test_std=('mse_test', 'std'),
        epochs_run_mean=('epochs_run', 'mean'),
        fit_time_sec_mean=('fit_time_sec', 'mean')
    ).reset_index()
dfi.export(df_optimizer.sort_values(by='mse_test_mean', ascending=True), "results_analysis/preprocessing_optimizers.png")

group_cols = ['structure', 'preprocessing']
df_structure = df_filtered.groupby(group_cols).agg(
        mse_train_mean=('mse_train', 'mean'),
        mse_train_std=('mse_train', 'std'),
        mse_test_mean=('mse_test', 'mean'),
        mse_test_std=('mse_test', 'std'),
        epochs_run_mean=('epochs_run', 'mean'),
        fit_time_sec_mean=('fit_time_sec', 'mean')
    ).reset_index()
dfi.export(df_structure.sort_values(by='mse_test_mean', ascending=True), "results_analysis/preprocessing_structures.png")
