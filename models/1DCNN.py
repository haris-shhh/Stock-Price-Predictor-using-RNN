HORIZON = 1
WINDOW_SIZE = 7
full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
len(full_windows), len(full_labels)

# Make train and testing windows
train_windows_scaled, test_windows_scaled, val_windows_scaled, train_labels, test_labels, val_labels = make_train_test_splits(windows=full_windows, labels=full_labels,test_split=0.2)
len(train_windows_scaled), len(test_windows_scaled), len(val_windows_scaled), len(train_labels), len(test_labels), len(val_labels)

# Before we pass our data to the Conv1D layer, we have to reshape it in order to make sure it works
x = tf.constant(train_windows_scaled[0])
expand_dims_layer = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)) # add an extra dimension for timesteps
print(f"Original shape: {x.shape}") # (WINDOW_SIZE)
print(f"Expanded shape: {expand_dims_layer(x).shape}") # (WINDOW_SIZE, input_dim)
print(f"Original values with expanded shape:\n {expand_dims_layer(x)}")

def build_model_conv1D(hp):
    model = Sequential([
        layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),  # Ensure input is correctly shaped
        layers.Conv1D(
            filters=hp.Int('filters', min_value=32, max_value=256, step=32),
            kernel_size=hp.Choice('kernel_size', values=[3, 5, 7]),
            padding="causal",
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(
                l1=hp.Float('l1', min_value=1e-5, max_value=1e-2, sampling='log'),
                l2=hp.Float('l2', min_value=1e-5, max_value=1e-2, sampling='log')
            )
        ),
        layers.Dense(1)
    ])
    model.compile(
        loss="mae",
        optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'))
    )
    return model

# Set seed for reproducibility
tf.random.set_seed(42)

# Create the Hyperband tuner
tuner = Hyperband(
    build_model_conv1D,
    objective="val_loss",
    max_epochs=70,  # Maximum number of epochs to train one model
    factor=3,        # Reduction factor for the number of epochs and number of models in each bracket
    executions_per_trial=3,  # How many times each model configuration will be run
    directory='hyperparameter_tuning',
    project_name='hyperband_tuning_CONV1D'
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
best_model_3 = tuner.hypermodel.build(best_hps)
# Fit the best model
best_model_3.fit(
    train_windows_scaled,
    train_labels,
    epochs=200,
    batch_size=128,
    validation_data=(val_windows_scaled, val_labels),
    callbacks=[early_stopping]
)

# Save the best model
best_model_3.save("/content/hyperparameter_tuning/hyperband_tuning_CONV1D/best_model_3.keras")

best_model_3_path = "/content/hyperparameter_tuning/hyperparameter_tuning_CONV1D/best_model_3.keras"
best_model_3 = tf.keras.models.load_model(best_model_3_path, safe_mode=False)

# Evaluate the loaded model on the test data
evaluation_result = best_model_3.evaluate(test_windows_scaled, test_labels)
print("Evaluation result (loss, metrics):", evaluation_result)

model_3_preds = make_preds(best_model_3, input_data=test_windows_scaled)

# Make predictions
model_3_results = evaluate_preds(y_true=tf.squeeze(test_labels), y_pred=model_3_preds)
model_3_results

plt.figure(figsize=(10,7))
# Plot the test data
plot_time_series(timesteps=X_test[-len(test_windows_scaled):], values=test_labels[:, 0],  label='Test Data', format="-", color='blue')
# Plot the mean of the Conv1D model's predictions
# Calculate the mean of the 7 prediction values across the second dimension using TensorFlow's reduce_mean
model_3_mean_preds = tf.reduce_mean(model_3_preds, axis=1)
plot_time_series(timesteps=X_test[-len(test_windows_scaled):], values=model_3_mean_preds,  format="-", label='Model 3', color='orange')
plt.legend()
plt.show()
