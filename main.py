
import numpy as np
from model import build_transformer_model
from utils import *

import math

all_sequences = np.load("all_sequences.npy")
all_labels = np.load("all_labels.npy")

# creating the validation and test set.

np.random.seed(42)

shuffled_indices = np.random.permutation(len(all_sequences))
all_sequences = all_sequences[shuffled_indices]
all_labels = all_labels[shuffled_indices]

train_size = int(len(all_sequences) * 0.9)

# split the sequences
train_sequences = all_sequences[:train_size]
train_labels = all_labels[:train_size]

other_sequences = all_sequences[train_size:]
other_labels = all_labels[train_size:]

shuffled_indices = np.random.permutation(len(other_sequences))

other_sequences = other_sequences[shuffled_indices]
other_labels = other_labels[shuffled_indices]

validation_size = int(len(other_sequences) * 0.5)

validation_sequences = other_sequences[:validation_size]
validation_labels = other_labels[:validation_size]

test_sequences = other_sequences[validation_size:]
test_labels = other_labels[validation_size:]


input_shape = train_sequences.shape[1:]
head_size = 256
num_heads = 16
ff_dim = 1024
num_layers = 12
dropout = 0.20

model = build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_layers, dropout)
model.summary()

BATCH_SIZE = 64  # Number of training examples used to calculate each iteration's gradient
EPOCHS = 10  # Total number of times the entire dataset is passed through the network

optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=custom_mae_loss, metrics=[dir_acc])


model.fit(
    train_sequences,  # Training features
    train_labels,  # Training labels
    validation_data=(validation_sequences, validation_labels),  # Validation data
    epochs=EPOCHS,  # Number of epochs to train for
    batch_size=BATCH_SIZE,  # Size of each batch
    shuffle=True,  # Shuffle training data before each epoch
    callbacks=[checkpoint_callback_train, checkpoint_callback_val, get_lr_callback(batch_size=BATCH_SIZE, epochs=EPOCHS)]  # Callbacks for saving models and adjusting learning rate
)