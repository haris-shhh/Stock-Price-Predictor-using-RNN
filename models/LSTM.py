HORIZON = 1 # predict next day
WINDOW_SIZE = 7 # use previous week worth of data

full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
len(full_windows), len(full_labels)

# Make train and testing windows
train_windows_scaled, test_windows_scaled, val_windows_scaled, train_labels, test_labels, val_labels = make_train_test_splits(windows=full_windows, labels=full_labels,test_split=0.2)
len(train_windows_scaled), len(test_windows_scaled), len(val_windows_scaled), len(train_labels), len(test_labels), len(val_labels)

# Define a function to build your model with hyperparameters
def build_model_LSTM(hp):
    # Define hyperparameters
    units = hp.Int('units', min_value=32, max_value=256, step=32)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # L1 and L2 regularization terms are used to prevent overfitting
    l1_value = hp.Float('l1', min_value=0.0001, max_value=0.01, sampling='log')  # Adjusted min and max values
    l2_value = hp.Float('l2', min_value=0.0001, max_value=0.01, sampling='log')  # Adjusted min and max values
    clip_value = hp.Float('clip_norm', min_value=0.1, max_value=1.0, step=0.1)
    # Define input layer
    inputs = tf.keras.Input(shape=(WINDOW_SIZE,))
    # Expand dimensions to match LSTM input requirement
    x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs) # expand input dimension to be compatible with LSTM

    # Define LSTM layer
    x = layers.LSTM(
        units=units,
        activation='relu',
        return_sequences=False,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_value, l2=l2_value)
    )(x)

    # Output layer
    outputs = layers.Dense(HORIZON)(x)
    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Compile model with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=clip_value
    )
    model.compile(
        loss="mae",
        optimizer=optimizer
    )
    return model

# Set seed for reproducibility
tf.random.set_seed(42)

# Create the Hyperband tuner
tuner = Hyperband(
    build_model_LSTM,
    objective="val_loss",
    max_epochs=70,  # Maximum number of epochs to train one model
    factor=3,        # Reduction factor for the number of epochs and number of models in each bracket
    executions_per_trial=3,  # How many times each model configuration will be run
    directory='hyperparameter_tuning',
    project_name='hyperband_tuning_model_LSTM'
)
# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
# Perform the hyperparameter search
tuner.search(
    train_windows_scaled,
    train_labels,
    epochs=70,
    validation_data=(val_windows_scaled, val_labels),
    callbacks=[early_stopping]
)
# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# Build the model with the best hyperparameters
best_model_LSTM = tuner.hypermodel.build(best_hps)
# Fit the best model
best_model_LSTM.fit(
    train_windows_scaled,
    train_labels,
    epochs=300,
    batch_size=128,
    validation_data=(val_windows_scaled, val_labels),
    callbacks=[early_stopping]
)
# Save the best model
best_model_LSTM.save("/content/hyperparameter_tuning/hyperband_tuning_model_LSTM/best_model_LSTM.keras")
best_model_LSTM_path = "/content/hyperparameter_tuning/hyperparameter_tuning_LSTM/best_model_LSTM.keras"
best_model_LSTM = tf.keras.models.load_model(best_model_LSTM_path, safe_mode=False)

# Evaluate the loaded model on the test data
evaluation_result = best_model_LSTM.evaluate(test_windows_scaled, test_labels)
print("Evaluation result (loss, metrics):", evaluation_result)
model_LSTM_preds = make_preds(best_model_LSTM, input_data=test_windows_scaled)

# Make predictions
model_LSTM_results = evaluate_preds(y_true=tf.squeeze(test_labels), y_pred=model_LSTM_preds)
model_LSTM_results

plt.figure(figsize=(10,7))
plot_time_series(timesteps=X_test[-len(test_windows_scaled):], values= test_labels[:, 0],label='Test Data',format="-", color='blue')
plot_time_series(timesteps=X_test[-len(test_windows_scaled):], values= model_LSTM_preds, format="-", label='Model_LSTM', color='Orange')
     
