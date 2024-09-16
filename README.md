# Stock-Price-Predictor-using-RNN

The prediction of stock prices has been a challenging and widely studied problem in the field of finance and data science. Stock price prediction is essential for financial analysts to make informed decisions and mitigate risks in the stock market. In this report, we will delve into the process of
building a stock price predictor using Yfinance data and machine learning techniques.
By leveraging historical stock price data and various machine learning algorithms, we aim to develop a robust predictor that can forecast future stock prices with reasonable accuracy. This report will provide a comprehensive overview of the data acquisition process, exploratory data
analysis, data preprocessing, model selection, training, evaluation, and potential challenges associated with stock price prediction.
This project aims to predict stock prices using advanced machine learning techniques, especially Long Short-Term Memory (LSTM) networks. LSTM networks are well-suited for this task as they can capture temporal dependencies and patterns. This approach is expected to provide more accurate predictions than traditional statistical methods.

## Project Structure

- **Stock_Price_Predictor.ipynb**: Jupyter Notebook containing the implementation of the stock price prediction models.
- **results/**: Directory to save trained models.
- **models/**: Directory containing scripts used for data processing and model training.

## Required Python packages:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - tensorflow

## Data
The yfinance dataset provides many values of the stock performance as discussed above. Since the ‘Close’ price of the stock is the most relevant value of the stock, I have created a new NumPy array with only the ‘Date’ and ‘Close’ price data. This was done to ensure the models libraries are compatible with the data.
The data does not have any losses and null values as yfinance is updated regularly hence no null handling needed to be done. The data was normalized using the Min- Max Scaler to ensure that the model receives data within a scaled range (0 to 1). 
This is helpful for algorithms that are sensitive to the scale of the input data, it preserves the shape of the original distribution and does not reduce the importance of outliers.
The data was then split in series into training, validation, and testing sets. Each of the sets serve their purposes:
- The training set is used by the model to learn the patterns in the data, this set is repeatedly presented to the model for optimization of the parameters to reduce forecasting errors.
- The validation set is used for model tuning, which involves iterating through different hyperparameter values.
- The test set is the unseen portion of the dataset that is reserved for final evaluation, this provides an unbiased evaluation of the ‘best model’ fit.
![image](https://github.com/user-attachments/assets/4beff27a-7c86-4a1a-8e6c-0fa5c27ebd4c)

The historical stock price data should be placed in the `data/` directory. Ensure the data is in a CSV format with appropriate columns, such as `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`.
![image](https://github.com/user-attachments/assets/1b063565-9351-4d71-96a5-b35800292201)


## Analysis

The model evaluation results show that LSTM has the lowest error statistic scores between the rest of the models and has performed best in stock price prediction.
However, we can also observe that DENSE 1 model also has a very low score, the reason for that would be the window size used. DENSE models are neural networks that are very good at fitting the models with sufficient number of neurons, but the model suffers from overfitting the training data and is very susceptible to noisy volatile stock price data.
We should also note when DENSE 1 performed on par with LSTM, DENSE 2 performs worse, this can be explained by the window size used in the models for prediction. While DENSE 1 had a window size of 7 with a horizon value of 1, DENSE 2 had window size 60 with horizon = 1. When training neural network models, beginning the training window too far back in time can make the outcomes more vulnerable to noise, as the testing data increased in size, the result was exposed to more volatility in the data. This is a problem with forecasting with neural networks.
1D-CNN, another neural network, performed worse than DENSE model due to backpropagation and the vanishing gradient problem, due to which the information from the previous layer is lost. 1D-CNN also suffers from sliding window problem, it can only use a fixed predetermined subset of data to make a prediction at each step.
Stock prices are non-stationary, meaning their statistical properties change over time, as a sliding window is being trained in a fixed window, that data might not be representative of trends outside the window, thus performing worse.
LSTM’s have the ability to filter out noise and are specifically designed to handle sequential data. They can also adjust to new patterns and utilize past information over long sequences, which is critical in stock price prediction.
![image](https://github.com/user-attachments/assets/cc30ae68-b4e5-407c-8ea0-e14167a8d6e0)

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any improvements or suggestions.

## License

This project is not licensed.

---
