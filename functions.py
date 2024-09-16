def evaluate_preds(y_true, y_pred):
    # Make sure our data is in float 32 data types
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    # Check if y_pred is multi-dimensional and handle accordingly
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
        # Assuming you want to take the mean of predictions or compare against a specific column
        y_pred = tf.reduce_mean(y_pred, axis=1)  # Taking mean across predictions, adjust if necessary
    # Calculate various evaluation metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    return {
        'MAE': mae.numpy(),
        'MSE': mse.numpy(),
        'RMSE': rmse.numpy(),
        'MAPE': mape.numpy()
    }
  
# Visualize our train and test data.
def plot_time_series(timesteps, values,color, format=".", start=0, end= None, label=None):
  plt.plot(timesteps[start:end], values[start:end],format, label=label,color=color)
  plt.xlabel('Year')
  plt.ylabel('Nvidia Stock Prices')
  if label:
    plt.legend(fontsize=10)
    
def get_labelled_window(x, horizon=1):
    return x[:, :-horizon], x[:, -horizon:]

# Testing the window labelling function:
test_windows, test_label = get_labelled_window(tf.expand_dims(tf.range(8), axis=0))
print(f"Window: {tf.squeeze(test_windows).numpy()} -> Label: {tf.squeeze(test_label).numpy()}")

def make_windows(x, window_size=7, horizon=1):
  #Turns a 1D array into a 2D array of sequential windows of window_size.
  # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
  window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
  # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
  window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T # create 2D array of windows of size window_size
  # 3. Index on the target array (time series) with 2D array of multiple window steps
  windowed_array = x[window_indexes]
  # 4. Get the labelled windows
  windows, labels = get_labelled_window(windowed_array, horizon=horizon)
  return windows, labels

def make_train_test_splits(windows, labels, test_split=0.2, validation_split=0.1):
    # Splitting data into train and test initially
    split_size = int(len(windows) * (1 - test_split))  # this will default to 80% train/20% test
    train_windows = windows[:split_size]
    test_windows = windows[split_size:]
    train_labels = labels[:split_size]
    test_labels = labels[split_size:]

    # Further split train data into train and validation sets
    validation_size = int(len(train_windows) * validation_split)  # 10% of the training set for validation
    val_windows = train_windows[-validation_size:]
    val_labels = train_labels[-validation_size:]

    # Update train data by removing validation data
    train_windows = train_windows[:-validation_size]
    train_labels = train_labels[:-validation_size]

    # Feature Scaling (fit only on training data)
    scaler = MinMaxScaler()
    train_windows_scaled = scaler.fit_transform(train_windows.reshape(-1, train_windows.shape[-1])).reshape(train_windows.shape)
    val_windows_scaled = scaler.transform(val_windows.reshape(-1, val_windows.shape[-1])).reshape(val_windows.shape)
    test_windows_scaled = scaler.transform(test_windows.reshape(-1, test_windows.shape[-1])).reshape(test_windows.shape)

    return train_windows_scaled, val_windows_scaled, test_windows_scaled, train_labels, val_labels, test_labels

def make_preds(model, input_data):
  forecast = model.predict(input_data)
  return tf.squeeze(forecast)
     
  
