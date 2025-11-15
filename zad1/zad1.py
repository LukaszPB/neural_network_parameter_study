import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pandas as pd

batch_sizes = [1,2,5,10,50,100,250,500,1000,5000,10000,20000,30000,40000,50000,60000]
losses = []
accuracies = []
times = []

class BatchLossLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.batch_losses = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs["loss"])

def train_model(batch_size):
    # deep learning library. Tensors are just multi-dimensional arrays
    mnist = tf.keras.datasets.mnist
    # mnist is a dataset of 28x28 images of handwritten digits and their labels
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    # unpacks images to x_train/x_test and labels to y_train/y_test
    x_train = tf.keras.utils.normalize(x_train, axis=1) # scales data between 0 and 1
    x_test = tf.keras.utils.normalize(x_test, axis=1) # scales data between 0 and 1
    model = tf.keras.models.Sequential() # a basic feed-forward model
    model.add(tf.keras.layers.Flatten()) # takes our 28x28 and makes it 1x784
    model.add(tf.keras.layers.Dense(128, activation= tf.keras.activations.relu))
    # a simple fully-connected layer, 128 units, relu activation
    model.add(tf.keras.layers.Dense(128, activation= tf.keras.activations.relu))
    # a simple fully-connected layer, 128 units, relu activation
    model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))
    # our output layer. 10 units for 10 classes.
    #Softmax for probability distribution
    model.summary()
    model.compile(optimizer='adam', # Good default optimizer to start with
    loss='sparse_categorical_crossentropy',
    # how will we calculate our "error." Neural network aims to minimize loss.
    metrics=['accuracy'])
    # what to track
    logger = BatchLossLogger()

    start_time = time.time()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=3, callbacks=[logger]) # train the model
    end_time = time.time()
    val_loss, val_acc = model.evaluate(x_test, y_test) # evaluate the out of sample data with model

    return logger.batch_losses, val_loss, val_acc, end_time-start_time

plt.figure(figsize=(12,7))

for batch_size in batch_sizes:
    batch_losses, val_loss, val_acc, train_time = train_model(batch_size)

    losses.append(val_loss)
    accuracies.append(val_acc)
    times.append(train_time)

    flat_losses = [float(x) for x in batch_losses]
    smooth = pd.Series(flat_losses).rolling(window=50, min_periods=1).mean()

    plt.plot(smooth, label=f"Batch size {batch_size}")

plt.xscale('log')   # czytelność przy różnych długościach
plt.xlabel("Iteracja (log)", fontsize=14)
plt.ylabel("Loss (wygładzony)", fontsize=14)
plt.title("Zmiana loss w zależności od batch size", fontsize=16)
plt.legend(fontsize=12)
plt.savefig("loss_vs_iteration.png", dpi=300)
plt.close()

print("\n")
print(f"{'Batch size':>12} | {'Loss':>10} | {'Accuracy':>10} | {'Time (s)':>10}")
print("-" * 50)

for i, batch_size in enumerate(batch_sizes):
    print(f"{batch_size:>12} | {losses[i]:>10.4f} | {accuracies[i]:>10.4f} | {times[i]:>10.2f}")
print("")

plt.figure(figsize=(8, 5))
plt.scatter(batch_sizes, losses, color='blue')
plt.xlabel("Batch size")
plt.ylabel("Loss")
plt.title("Loss vs Batch Size")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_vs_batch.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(batch_sizes, accuracies, color='blue')
plt.xlabel("Batch size")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Batch Size")
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_vs_batch.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(batch_sizes, times, color='blue')
plt.xlabel("Batch size")
plt.ylabel("Time [s]")
plt.title("Time vs Batch Size")
plt.grid(True)
plt.tight_layout()
plt.savefig("time_vs_batch.png", dpi=300)
plt.close()