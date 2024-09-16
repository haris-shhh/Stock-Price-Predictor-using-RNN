# DENSE model that predicts 1 step ahead with 7 days training data

HORIZON = 1 # predict one step at a time
WINDOW_SIZE = 7 # use 7 timesteps in the past
full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
len(full_windows), len(full_labels)

# Make train and testing windows
train_windows_scaled, test_windows_scaled, val_windows_scaled, train_labels, test_labels, val_labels = make_train_test_splits(windows=full_windows, labels=full_labels, test_split=0.2)
len(train_windows_scaled), len(test_windows_scaled), len(val_windows_scaled), len(train_labels), len(test_labels), len(val_labels)

def create_model_1(hp):
    # Define hyperparameters
    units = hp.Int('units', min_value=32, max_value=256, step=32)
    activation = hp.Choice('activation', ['relu', 'tanh', 'elu'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    l1_reg = hp.Float('l1_reg', min_value=1e-5, max_value=1e-2, sampling='log')
    l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='log')

    # Construct model
    model = Sequential([
        layers.Dense(units, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
        layers.Dense(1, activation="linear")  # Dynamic output size
    ], name="model_1_dense")

    # Define optimizer (using only Adam optimizer)
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    # Compile model
    model.compile(loss="mae", optimizer=optimizer, metrics=["mae"])

    return model

# Set seed for reproducibility
tf.random.set_seed(42)

# Create the Hyperband tuner
tuner = Hyperband(
    create_model_1,
    objective="val_loss",
    max_epochs=70,  # Maximum number of epochs to train one model
    factor=3,        # Reduction factor for the number of epochs and number of models in each bracket
    executions_per_trial=3,  # How many times each model configuration will be run
    directory='hyperparameter_tuning',
    project_name='hyperband_tuning_model_1'
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
best_model_1 = tuner.hypermodel.build(best_hps)

# Fit the best model
best_model_1.fit(
    train_windows_scaled,
    train_labels,
    epochs=200,
    batch_size=128,
    validation_data=(val_windows_scaled, val_labels),
    callbacks=[early_stopping]
)

# Save the best model
best_model_1.save("/content/hyperparameter_tuning/hyperband_tuning_model_1/best_model_1.keras")

# Load the best model
best_model_path = "/content/hyperparameter_tuning/hyperband_tuning_model_1/best_model_1.keras"
best_model = tf.keras.models.load_model(best_model_path)

# Evaluate the loaded model on the test data
evaluation_result = best_model.evaluate(test_windows_scaled, test_labels)
print("Evaluation result (loss, metrics):", evaluation_result)

model_1_preds = make_preds(best_model, input_data=test_windows_scaled)

model_1_results = evaluate_preds(y_true=tf.squeeze(test_labels), y_pred=model_1_preds)
model_1_results

plt.figure(figsize=(10,7))

plot_time_series(timesteps=X_test[-len(test_windows_scaled):], values= test_labels[:, 0],label='Test Data',format="-", color='blue')
plot_time_series(timesteps=X_test[-len(test_windows_scaled):], values= model_1_preds, format="-", label='DENSE_1', color='Orange')

# DENSE model that predicts 1 day in the future with 60 days training data

HORIZON = 1 # predict one step at a time
WINDOW_SIZE = 60 # use 60 timesteps in the past
full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
len(full_windows), len(full_labels)
# Make train and testing windows
train_windows_scaled, test_windows_scaled, val_windows_scaled, train_labels, test_labels, val_labels = make_train_test_splits(windows=full_windows, labels=full_labels,test_split=0.2)
len(train_windows_scaled), len(test_windows_scaled), len(val_windows_scaled), len(train_labels), len(test_labels), len(val_labels)

def create_model_2(hp):
    # Define hyperparameters
    units = hp.Int('units', min_value=32, max_value=256, step=32)
    activation = hp.Choice('activation', ['relu', 'tanh', 'elu'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    l1 = hp.Float('l1_reg', min_value=1e-5, max_value=1e-2, sampling='log')
    l2 = hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='log')

    # Construct model
    model = tf.keras.Sequential([
        layers.Dense(units, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
        layers.Dense(HORIZON, activation="linear")
    ], name="model_2")

    # Define optimizer (using only Adam optimizer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile model
    model.compile(loss="mae", optimizer=optimizer, metrics=["mae"])

    return model

  # Set seed for reproducibility
tf.random.set_seed(42)
# Create the Hyperband tuner
tuner = Hyperband(
    create_model_2,
    objective="val_loss",
    max_epochs=70,  # Maximum number of epochs to train one model
    factor=3,        # Reduction factor for the number of epochs and number of models in each bracket
    executions_per_trial=3,  # How many times each model configuration will be run
    directory='hyperparameter_tuning',
    project_name='hyperband_tuning_model_2'
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
best_model_2 = tuner.hypermodel.build(best_hps)
# Fit the best model
best_model_2.fit(
    train_windows_scaled,
    train_labels,
    epochs=200,
    batch_size=128,
    validation_data=(val_windows_scaled, val_labels),
    callbacks=[early_stopping]
)

# Save the best model
best_model_2.save("/content/hyperparameter_tuning/hyperband_tuning_model_2/best_model_2.keras")
best_model_2_path = "/content/hyperparameter_tuning/hyperband_tuning_model_2/best_model_2.keras"
best_model_2 = tf.keras.models.load_model(best_model_2_path)

# Evaluate the loaded model on the test data
evaluation_result = best_model_2.evaluate(test_windows_scaled, test_labels)
print("Evaluation result (loss, metrics):", evaluation_result)

model_2_preds = make_preds(best_model_2, input_data=test_windows_scaled)
model_2_results = evaluate_preds(y_true=tf.squeeze(test_labels), y_pred=model_2_preds)
model_2_results

plt.figure(figsize=(10,7))
plot_time_series(timesteps=X_test[-len(test_windows_scaled):], values= test_labels[:, 0],label='Test Data',format="-", color='blue')
plot_time_series(timesteps=X_test[-len(test_windows_scaled):], values= model_2_preds, format="-", label='DENSE_2', color='Orange')
