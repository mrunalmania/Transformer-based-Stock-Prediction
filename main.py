import numpy as np

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

