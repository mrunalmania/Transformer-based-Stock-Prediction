# Stock Prediction using Transformer Architecture Neural Network

This project focuses on stock price prediction using a transformer-based architecture. The model is trained on stock data retrieved via the `yfinance` library, applying various technical indicators to enhance predictive accuracy.

## Stock Data Used for Training
The model is trained on data from the following stocks:
- Apple (AAPL)
- Microsoft (MSFT)
- Amazon (AMZN)
- Meta (META)
- Google (GOOGL)

The dataset includes stock data from the last 30 days, sampled at 5-minute intervals. Additionally, several technical indicators, such as RSI (Relative Strength Index), Bollinger Bands, and ROC (Rate of Change), were calculated and added to the dataset.

## Dataset Preparation
The dataset was pre-processed and stored in a NumPy array for training.

## Challenges
1. **TensorFlow LayerNormalization Issue**: In TensorFlow 2.16, I encountered an issue with the `LayerNormalization` layer. This was resolved by uninstalling TensorFlow and upgrading to version 2.17.
2. **Hardware Limitations**: Due to limited computing resources, I was unable to train the model for more than 10 epochs.

## Results
The model achieved a directional accuracy of 70% on both the training and validation sets. The overall training accuracy stands at 65%.

## Future Work
In the future, I plan to:
- Transition from TensorFlow to PyTorch to leverage its flexibility.
- Add more layers to the transformer architecture, making it more complex to better capture the intricacies of stock market data.

## Credits
- **Author**: Mrunal Ashwinbhai Mania
- **University**: Arizona State University
- **Email**: mrunal29mania@outlook.com
