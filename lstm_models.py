# =============================================================================
# STEP 2: MODEL TRAINING & EVALUATION
# PJM Energy Forecasting — RNN, LSTM, Bi-LSTM, Encoder-Decoder
# =============================================================================

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")


# =============================================================================
# LOAD DATA  (saved by data_preparation.py)
# =============================================================================
print("\nLoading preprocessed data...")
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val   = np.load('X_val.npy')
y_val   = np.load('y_val.npy')
X_test  = np.load('X_test.npy')
y_test  = np.load('y_test.npy')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Shorthand dimensions — used when building every model
TIMESTEPS  = X_train.shape[1]   # 168  (lookback window)
N_FEATURES = X_train.shape[2]   # 11   (number of input features)
HORIZON    = y_train.shape[1]   # 24   (hours to predict)
TARGET_IDX = 0                  # PJME is the first feature

print(f"  Input shape:  (samples={len(X_train):,}, timesteps={TIMESTEPS}, features={N_FEATURES})")
print(f"  Output shape: (samples={len(y_train):,}, horizon={HORIZON})")


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def build_rnn():
    """Simple RNN — the most basic sequential model (our baseline)."""
    model = models.Sequential(name='SimpleRNN')
    model.add(layers.SimpleRNN(64, return_sequences=True, input_shape=(TIMESTEPS, N_FEATURES)))
    model.add(layers.Dropout(0.2))
    model.add(layers.SimpleRNN(32))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(HORIZON))
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
    return model


def build_lstm():
    """LSTM — adds memory gates so the model remembers long-term patterns."""
    model = models.Sequential(name='LSTM')
    model.add(layers.LSTM(128, return_sequences=True, input_shape=(TIMESTEPS, N_FEATURES)))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(64))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(HORIZON))
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
    return model


def build_bilstm():
    """Bidirectional LSTM — reads the sequence forwards AND backwards."""
    model = models.Sequential(name='BiLSTM')
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True),
                                   input_shape=(TIMESTEPS, N_FEATURES)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Bidirectional(layers.LSTM(64)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(HORIZON))
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
    return model


def build_encoder_decoder():
    """
    Encoder-Decoder LSTM — two-part model:
      Encoder: reads the 168-hour input and compresses it into a summary
      Decoder: uses that summary to generate the 24-hour forecast
    """
    # -- Encoder --
    enc_input  = layers.Input(shape=(TIMESTEPS, N_FEATURES))
    enc_out, h, c = layers.LSTM(128, return_sequences=False,
                                 return_state=True)(enc_input)
    enc_out    = layers.Dropout(0.2)(enc_out)

    # -- Bridge: repeat the encoder summary once per output step --
    repeated   = layers.RepeatVector(HORIZON)(enc_out)

    # -- Decoder --
    dec_out    = layers.LSTM(64, return_sequences=True)(repeated,
                              initial_state=[h, c])
    dec_out    = layers.Dropout(0.2)(dec_out)
    dec_out    = layers.LSTM(32, return_sequences=True)(dec_out)
    output     = layers.TimeDistributed(layers.Dense(1))(dec_out)
    output     = layers.Flatten()(output)   # shape → (batch, 24)

    model = models.Model(enc_input, output, name='EncoderDecoder')
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
    return model


# =============================================================================
# HELPERS: train, evaluate, plot
# =============================================================================

def train(model, epochs=50, batch_size=64):
    """Train a model with early stopping and learning-rate reduction."""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(f'best_{model.name}.keras', monitor='val_loss',
                        save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
    ]
    print(f"\n--- Training {model.name} ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def evaluate(model):
    """
    Run the model on the test set and return RMSE, MAE, MAPE.
    Also converts predictions back from 0–1 scale to real MW values.
    """
    y_pred_scaled = model.predict(X_test, verbose=0)

    # --- inverse-transform: scaled → real MW ---
    # We create a dummy array the same width as our feature set,
    # put our values in the right column, then unscale everything.
    n_feat = scaler.n_features_in_

    def unscale(arr):
        dummy = np.zeros((arr.shape[0] * arr.shape[1], n_feat))
        dummy[:, TARGET_IDX] = arr.flatten()
        return scaler.inverse_transform(dummy)[:, TARGET_IDX].reshape(arr.shape)

    y_pred = unscale(y_pred_scaled)
    y_true = unscale(y_test)

    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    mae  = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    mape = mean_absolute_percentage_error(y_true.flatten(), y_pred.flatten()) * 100

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape,
            'y_true': y_true, 'y_pred': y_pred}


def plot_history(history, name):
    """Save a training-curve plot (loss and MAE over epochs)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, metric, label in zip(axes,
                                  ['loss', 'mae'],
                                  ['MSE Loss', 'MAE']):
        ax.plot(history.history[metric],     label='Train')
        ax.plot(history.history[f'val_{metric}'], label='Validation')
        ax.set_title(f'{name} — {label}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'training_history_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_predictions(y_true, y_pred, name, n_samples=5):
    """Save a plot showing predicted vs actual for 5 random test samples."""
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3 * n_samples))
    for i in range(n_samples):
        idx = np.random.randint(0, len(y_true))
        axes[i].plot(y_true[idx], label='Actual',    marker='o', linewidth=2)
        axes[i].plot(y_pred[idx], label='Predicted', marker='x', linewidth=2)
        axes[i].set_title(f'Sample {i+1} — 24-Hour Forecast', fontweight='bold')
        axes[i].set_xlabel('Hour ahead')
        axes[i].set_ylabel('MW')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'predictions_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN — train all 4 models, collect results
# =============================================================================
all_models = {
    'SimpleRNN':      build_rnn,
    'LSTM':           build_lstm,
    'BiLSTM':         build_bilstm,
    'EncoderDecoder': build_encoder_decoder,
}

results = {}

for name, build_fn in all_models.items():
    model   = build_fn()
    history = train(model)

    plot_history(history, name)

    metrics = evaluate(model)
    results[name] = metrics

    plot_predictions(metrics['y_true'], metrics['y_pred'], name)

    print(f"\n{name} results:")
    print(f"  RMSE: {metrics['RMSE']:,.1f} MW")
    print(f"  MAE:  {metrics['MAE']:,.1f} MW")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")


# =============================================================================
# COMPARISON TABLE & CHART
# =============================================================================
print("\n" + "="*50)
print("FINAL COMPARISON")
print("="*50)

comparison = pd.DataFrame([
    {'Model': name,
     'RMSE (MW)': f"{m['RMSE']:.1f}",
     'MAE (MW)':  f"{m['MAE']:.1f}",
     'MAPE (%)':  f"{m['MAPE']:.2f}"}
    for name, m in results.items()
])
print(comparison.to_string(index=False))
comparison.to_csv('model_comparison.csv', index=False)

# Bar chart comparing all 3 metrics
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for ax, metric in zip(axes, ['RMSE', 'MAE', 'MAPE']):
    values = [results[n][metric] for n in all_models]
    ax.bar(all_models.keys(), values, color=colors)
    ax.set_title(f'{metric} Comparison', fontweight='bold')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll done! Files saved:")
print("  • training_history_[model].png  (4 files)")
print("  • predictions_[model].png       (4 files)")
print("  • best_[model].keras            (4 files)")
print("  • model_comparison.png")
print("  • model_comparison.csv")