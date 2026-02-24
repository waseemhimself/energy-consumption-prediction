# ⚡ Energy Demand Forecasting with LSTM

A deep learning project for predicting energy demand using RNN, LSTM, Bidirectional LSTM, and Encoder-Decoder architectures on the PJM Interconnection hourly energy dataset.

##  Overview

This project implements and compares four different sequential neural network architectures for time series forecasting of energy demand:
- Simple RNN (baseline)
- LSTM (Long Short-Term Memory)
- Bidirectional LSTM
- Encoder-Decoder LSTM

The models predict 24-hour ahead energy demand using 168 hours (1 week) of historical data across 5 regions in the PJM Interconnection grid.

##  Features

- **Data Preprocessing Pipeline**: Automated cleaning, feature engineering, and sequence generation
- **Multiple Model Architectures**: Compare 4 different sequential models
- **Interactive Web Interface**: Streamlit-based UI for real-time predictions
- **Comprehensive Evaluation**: RMSE, MAE, MAPE metrics with visualizations
- **Model Persistence**: Save and load trained models for future use
- **Temporal Feature Engineering**: Cyclical encoding for time-based patterns

##  Dataset

**Source**: PJM Interconnection Hourly Energy Consumption Data (2005-2018)

**Regions**:
- PJME (PJM East)
- PJMW (PJM West)
- DAYTON
- AEP (American Electric Power)
- DUQ (Duquesne Light Company)

**Size**: 119,066 hourly observations after cleaning

**Features**: 11 (5 energy regions + 6 temporal features)


##  Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/energy-demand-forecasting.git
cd energy-demand-forecasting
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

##  Usage

### Step 1: Data Preparation

Run the data preparation script to clean and process the raw data:

```bash
python data_preparation.py
```

**Output**:
- `pjm_cleaned.csv` - Cleaned dataset
- `pjm_featured.csv` - Dataset with time features
- `X_train.npy`, `y_train.npy`, etc. - Processed sequences
- `scaler.pkl` - Normalization scaler
- 5 exploratory visualization plots

### Step 2: Train Models

Open and run the Jupyter notebook:

```bash
jupyter notebook lstm_models.ipynb
```

Or use the Python script:

```bash
python lstm_models.py
```

**Output**:
- `saved_models/` - Trained model files (.keras)
- Training history plots
- Prediction comparison plots
- `model_comparison.csv` - Performance metrics

**Training time**: ~20-40 minutes (depending on hardware)

### Step 3: Run Web Interface

Launch the interactive Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`
\
##  Models

### 1. Simple RNN
**Architecture**: 2 SimpleRNN layers (64, 32 units)  
**Purpose**: Baseline model for comparison

### 2. LSTM
**Architecture**: 2 LSTM layers (128, 64 units)  
**Purpose**: Standard approach for time series forecasting  
**Best for**: General-purpose forecasting

### 3. Bidirectional LSTM
**Architecture**: 2 Bidirectional LSTM layers (128, 64 units)  
**Purpose**: Captures forward and backward temporal patterns  
**Best for**: When context from both directions matters

### 4. Encoder-Decoder LSTM
**Architecture**: Encoder (128→64 units) + Decoder (64→32 units)  
**Purpose**: Sequence-to-sequence learning  
**Best for**: Longer forecast horizons

All models include:
- Dropout layers (0.2) for regularization
- Dense output layer for 24-hour predictions
- Adam optimizer
- MSE loss function

##  Results

Expected performance on test data:

| Model | RMSE (MW) | MAE (MW) | MAPE (%) |
|-------|-----------|----------|----------|
| Simple RNN | ~2,500 | ~1,800 | 6-8% |
| **LSTM** | **~1,800** | **~1,300** | **4-5%** |
| **Bi-LSTM** | **~1,700** | **~1,250** | **4-5%** |
| Encoder-Decoder | ~1,900 | ~1,400 | 4.5-6% |

*LSTM and Bi-LSTM typically provide the best performance.*

**Available date range**: 2017-03-24 to 2018-08-03 (test period)

## 📁 Project Structure

```
energy-demand-forecasting/
│
├── data_preparation.py          # Data cleaning and preprocessing
├── lstm_models.ipynb            # Model training notebook
├── lstm_models.py               # Model training script
├── app.py                       # Streamlit web interface
│
├── pjm_hourly_est.csv          # Raw dataset (add this file)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore rules
│
├── saved_models/                # Trained models (created during training)
│   ├── LSTM.keras
│   ├── BiLSTM.keras
│   ├── SimpleRNN.keras
│   └── EncoderDecoder.keras
│
├── pjm_cleaned.csv             # Cleaned data (created by data_preparation)
├── pjm_featured.csv            # Featured data (created by data_preparation)
├── scaler.pkl                  # Normalization scaler
├── X_train.npy, y_train.npy... # Processed sequences
│
└── outputs/                     # Generated plots and results
    ├── 01_time_series_overview.png
    ├── 02_correlation_matrix.png
    ├── 03_hourly_pattern.png
    ├── 04_weekly_pattern.png
    ├── 05_monthly_pattern.png
    ├── training_history_*.png
    ├── predictions_*.png
    └── model_comparison.png
```

## 🔧 Configuration

### Adjustable Parameters

In `data_preparation.py`:
```python
LOOKBACK = 168    # Hours of history to use (1 week)
HORIZON = 24      # Hours to predict ahead (1 day)
TARGET_COL = 'PJME'  # Which region to forecast
```

In `lstm_models.ipynb`:
```python
epochs = 50       # Maximum training epochs
batch_size = 64   # Batch size for training
```

---

⭐ If you found this project helpful, please consider giving it a star!
