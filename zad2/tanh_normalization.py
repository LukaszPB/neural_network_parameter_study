import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

df = pd.read_csv("ps6_dane1_ucz.txt", sep="\t")
x_train1_raw = df["x"].values.reshape(-1, 1)
y_train1 = df["y"].values.reshape(-1, 1)

df = pd.read_csv("ps6_dane1_test.txt", sep="\t")
x_test1_raw = df["x"].values.reshape(-1, 1)
y_test1 = df["y"].values.reshape(-1, 1)

x_train1_raw, x_val1_raw, y_train1, y_val1 = train_test_split(x_train1_raw, y_train1, test_size=0.2, random_state=42)

scaler_x = StandardScaler()
x_train1 = scaler_x.fit_transform(x_train1_raw) 
x_val1 = scaler_x.transform(x_val1_raw)
x_test1 = scaler_x.transform(x_test1_raw)

x_train1_LeCun = (x_train1_raw - x_train1_raw.mean(axis=0)) / (x_train1_raw.std(axis=0) * 0.577)
x_val1_LeCun = (x_val1_raw - x_val1_raw.mean(axis=0)) / (x_val1_raw.std(axis=0) * 0.577)
x_test1_LeCun = (x_test1_raw - x_test1_raw.mean(axis=0)) / (x_test1_raw.std(axis=0) * 0.577)


scaler_minmax = MinMaxScaler(feature_range=(-1, 1))
x_train1_mm = scaler_minmax.fit_transform(x_train1_raw)
x_val1_mm   = scaler_minmax.transform(x_val1_raw)
x_test1_mm  = scaler_minmax.transform(x_test1_raw)

print("=== StandardScaler ===")
print("Train: min =", np.min(x_train1), "max =", np.max(x_train1), "mean =", np.mean(x_train1))
print("Val:   min =", np.min(x_val1),   "max =", np.max(x_val1),   "mean =", np.mean(x_val1))
print("Test:  min =", np.min(x_test1),  "max =", np.max(x_test1),  "mean =", np.mean(x_test1))

print("\n=== LeCun ===")
print("Train: min =", np.min(x_train1_LeCun), "max =", np.max(x_train1_LeCun), "mean =", np.mean(x_train1_LeCun))
print("Val:   min =", np.min(x_val1_LeCun),   "max =", np.max(x_val1_LeCun),   "mean =", np.mean(x_val1_LeCun))
print("Test:  min =", np.min(x_test1_LeCun),  "max =", np.max(x_test1_LeCun),  "mean =", np.mean(x_test1_LeCun))

print("\n=== MinMaxScaler ===")
print("Train: min =", np.min(x_train1_mm), "max =", np.max(x_train1_mm), "mean =", np.mean(x_train1_mm))
print("Val:   min =", np.min(x_val1_mm),   "max =", np.max(x_val1_mm),   "mean =", np.mean(x_val1_mm))
print("Test:  min =", np.min(x_test1_mm),  "max =", np.max(x_test1_mm),  "mean =", np.mean(x_test1_mm))

print("\n=== Raw data ===")
print("Train: min =", np.min(x_train1_raw), "max =", np.max(x_train1_raw), "mean =", np.mean(x_train1_raw))
print("Val:   min =", np.min(x_val1_raw),   "max =", np.max(x_val1_raw),   "mean =", np.mean(x_val1_raw))
print("Test:  min =", np.min(x_test1_raw),  "max =", np.max(x_test1_raw),  "mean =", np.mean(x_test1_raw), "\n")

# Lista konfiguracji
activations = ['tanh']
optimizers = {
     "Adam": lambda: tf.keras.optimizers.Adam()
}
structures = [
    [64, 64, 32]
]

batch_size = 10
patience = 10

datasets = [
    ("Dane1_standardise", x_train1, y_train1, x_val1, y_val1, x_test1, y_test1),
    ("Dane1_LeCun", x_train1_LeCun, y_train1, x_val1_LeCun, y_val1, x_test1_LeCun, y_test1),
    ("Dane1_min-max", x_train1_mm, y_train1, x_val1_mm, y_val1, x_test1_mm, y_test1),
    ("Dane1_raw", x_train1_raw, y_train1, x_val1_raw, y_val1, x_test1_raw, y_test1),
]

os.makedirs("tanh_normalziation_result", exist_ok=True)

for act in activations:
    for opt_name, opt_fn in optimizers.items():
        for struct in structures:
            fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 24))
            fig.suptitle(f"Wyniki dla: {act}, {opt_name}, {struct}", fontsize=20)

            for idx, (dname, x_train, y_train, x_val, y_val, x_test, y_test) in enumerate(datasets):
                best_mse_test = float('inf')
                best_history = None
                best_pred = None
                for run_number in range(2):
                    model = tf.keras.models.Sequential()
                    for i, units in enumerate(struct):
                        if i == 0:
                            model.add(tf.keras.layers.Dense(units, activation=act, input_shape=(x_train.shape[1],)))
                        else:
                            model.add(tf.keras.layers.Dense(units, activation=act))
                    model.add(tf.keras.layers.Dense(1))

                    model.compile(optimizer=opt_fn(), loss='mse', metrics=['mse'])

                    early_stop = tf.keras.callbacks.EarlyStopping(
                        monitor='val_mse', patience=patience, restore_best_weights=True, mode='min'
                    )

                    start_time = time.time()
                    history = model.fit(
                        x_train, y_train,
                        validation_data=(x_val, y_val),
                        batch_size=batch_size,
                        epochs=1000,
                        callbacks=[early_stop],
                        verbose=0
                    )
                    fit_time = time.time() - start_time

                    y_pred = model.predict(x_test)

                    # Wyniki
                    mse_train = model.evaluate(x_train, y_train, verbose=0)[1]
                    mse_val = model.evaluate(x_val, y_val, verbose=0)[1]
                    mse_test = model.evaluate(x_test, y_test, verbose=0)[1]

                    epochs_run = history.epoch[-1] + 1

                    if best_mse_test > mse_test:
                        best_mse_test = mse_test
                        best_history = history
                        best_pred = y_pred

                # Wykresy
                ax_pred = axes[idx, 0]
                ax_mse = axes[idx, 1]

                if x_test.shape[1] == 1:
                    ax_pred.scatter(x_test, y_test, label="Rzeczywiste", s=10, alpha=0.7)
                    ax_pred.scatter(x_test, best_pred, label="Prognozowane", color='red', s=10, alpha=0.7)
                    ax_pred.set_xlabel('x', fontsize=14)
                    ax_pred.set_ylabel('y', fontsize=14)
                elif x_test.shape[1] == 2:
                    fig.delaxes(ax_pred)
                    ax_pred = fig.add_subplot(4, 2, idx*2 + 1, projection='3d')
                    ax_pred.scatter(x_test[:, 0], x_test[:, 1], y_test.flatten(), label="Rzeczywiste", s=10, alpha=0.7)
                    ax_pred.scatter(x_test[:, 0], x_test[:, 1], best_pred.flatten(), label="Prognozowane", color='red', s=10, alpha=0.7)
                    ax_pred.set_xlabel('x1', fontsize=14)
                    ax_pred.set_ylabel('x2', fontsize=14)
                    ax_pred.set_zlabel('y', fontsize=14)
                    ax_pred.tick_params(axis='x', labelsize=12)
                    ax_pred.tick_params(axis='y', labelsize=12)
                    ax_pred.tick_params(axis='z', labelsize=12)

                ax_pred.set_title(f"{dname} - Predykcje", fontsize=16)
                ax_pred.legend(fontsize=14)
                ax_pred.tick_params(axis='both', labelsize=14)

                ax_mse.plot(best_history.history['mse'], label='MSE treningowe')
                ax_mse.plot(best_history.history['val_mse'], label='MSE walidacyjne')
                ax_mse.set_xlabel('Epoki', fontsize=14)
                ax_mse.set_ylabel('MSE', fontsize=14)
                ax_mse.set_title(f"{dname} - MSE", fontsize=16)
                ax_mse.legend(fontsize=14)
                ax_mse.tick_params(axis='both', labelsize=14)

            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.savefig(f"tanh_normalziation_result/{act}-{opt_name}-{struct}.png")
            plt.close(fig)