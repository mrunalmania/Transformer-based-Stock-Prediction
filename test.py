from main import model
from main import test_sequences, test_labels


model.load_weights("transformer_val_model.keras")  # Load the best model from the validation phase
accuracy = model.evaluate(test_sequences, test_labels)[1]  # Evaluate the model on the test data
print(accuracy)

from sklearn.metrics import r2_score

predictions = model.predict(test_sequences)  # Make predictions on the test dataset
r2 = r2_score(test_labels[:, 1], predictions[:, 0])  # Calculate R-squared value
print(f"R-squared: {r2}")