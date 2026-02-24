# =============================================================================
# STEP 1: DATA PREPARATION
# PJM Energy Forecasting — Top 5 Regions
# =============================================================================

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

RAW_DATA_PATH = 'pjm_hourly_est.csv'
TARGET_COL    = 'PJME'   # the region we want to forecast
LOOKBACK      = 168      # how many past hours the model sees (1 week)
HORIZON       = 24       # how many future hours to predict (1 day)

ENERGY_COLS = ['PJME', 'PJMW', 'DAYTON', 'AEP', 'DUQ']


# =============================================================================
# 1. LOAD & CLEAN RAW DATA
# =============================================================================
print("Loading raw data...")
df = pd.read_csv(RAW_DATA_PATH)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)
print(f"  Raw shape: {df.shape}  →  {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"  Date range: {df['Datetime'].min().date()} to {df['Datetime'].max().date()}")

# Show missing values before cleaning
print("\n  Missing values per column (before cleaning):")
for col in df.columns[1:]:
    missing_pct = df[col].isna().mean() * 100
    print(f"    {col:<12} {missing_pct:.1f}% missing")

# Keep only the top 5 most complete regions + Datetime
print(f"\nKeeping top 5 regions: {ENERGY_COLS}")
df = df[['Datetime'] + ENERGY_COLS]

# Drop any row that has a missing value in any of the 5 columns
df = df.dropna().reset_index(drop=True)

print(f"  Clean shape: {df.shape}  →  {len(df):,} rows remaining")
print(f"  Date range after cleaning: {df['Datetime'].min().date()} to {df['Datetime'].max().date()}")
print(f"  Missing values remaining: {df.isnull().sum().sum()}")

# Save the cleaned data as a CSV
df.to_csv('pjm_cleaned.csv', index=False)
print("  Cleaned data saved to pjm_cleaned.csv")


# =============================================================================
# 2. ADD TIME FEATURES
# =============================================================================
print("\nAdding time features...")

# Basic time info
df['Hour']      = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek   # 0=Monday, 6=Sunday
df['Month']     = df['Datetime'].dt.month

# Cyclical encoding — makes hour 23 and hour 0 "close" to each other
df['Hour_sin']  = np.sin(2 * np.pi * df['Hour']  / 24)
df['Hour_cos']  = np.cos(2 * np.pi * df['Hour']  / 24)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# Weekend flag
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# All features the model will use as input
FEATURES = ENERGY_COLS + ['Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos',
                           'IsWeekend', 'DayOfWeek']
print(f"  Total features: {len(FEATURES)}: {FEATURES}")

# Save the data with all time features added as a CSV
df[['Datetime'] + FEATURES].to_csv('pjm_featured.csv', index=False)
print("  Featured data saved to pjm_featured.csv")


# =============================================================================
# 3. VISUALISATIONS
# =============================================================================
print("\nCreating plots...")

# Plot 1 — Energy demand over time for each region
fig, axes = plt.subplots(5, 1, figsize=(15, 12))
for i, col in enumerate(ENERGY_COLS):
    axes[i].plot(df['Datetime'], df[col], linewidth=0.5, alpha=0.7)
    axes[i].set_title(f'{col} Energy Demand Over Time', fontweight='bold')
    axes[i].set_ylabel('MW')
    axes[i].grid(True, alpha=0.3)
axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig('01_time_series_overview.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2 — How correlated are the regions with each other?
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df[ENERGY_COLS].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            square=True, ax=ax)
ax.set_title('Correlation Between Regions', fontweight='bold')
plt.tight_layout()
plt.savefig('02_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3 — Average demand by hour of day
fig, ax = plt.subplots(figsize=(12, 5))
for col in ENERGY_COLS:
    ax.plot(df.groupby('Hour')[col].mean(), marker='o', label=col)
ax.set_title('Average Demand by Hour of Day', fontweight='bold')
ax.set_xlabel('Hour')
ax.set_ylabel('Average MW')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('03_hourly_pattern.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 4 — Average demand by day of week
day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
fig, ax = plt.subplots(figsize=(12, 5))
for col in ENERGY_COLS:
    ax.plot(day_labels, df.groupby('DayOfWeek')[col].mean(), marker='o', label=col)
ax.set_title('Average Demand by Day of Week', fontweight='bold')
ax.set_xlabel('Day')
ax.set_ylabel('Average MW')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('04_weekly_pattern.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 5 — Average demand by month
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
fig, ax = plt.subplots(figsize=(12, 5))
for col in ENERGY_COLS:
    ax.plot(month_labels, df.groupby('Month')[col].mean(), marker='o', label=col)
ax.set_title('Average Demand by Month', fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Average MW')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('05_monthly_pattern.png', dpi=150, bbox_inches='tight')
plt.close()

print("  Saved 5 plots")


# =============================================================================
# 4. SPLIT DATA  (80% train | 10% validation | 10% test)
# =============================================================================
print("\nSplitting data...")
n = len(df)
train_end = int(n * 0.8)
val_end   = int(n * 0.9)

train_df = df[:train_end]
val_df   = df[train_end:val_end]
test_df  = df[val_end:]

print(f"  Train:      {len(train_df):,} rows  ({train_df['Datetime'].min().date()} to {train_df['Datetime'].max().date()})")
print(f"  Validation: {len(val_df):,} rows  ({val_df['Datetime'].min().date()} to {val_df['Datetime'].max().date()})")
print(f"  Test:       {len(test_df):,} rows  ({test_df['Datetime'].min().date()} to {test_df['Datetime'].max().date()})")


# =============================================================================
# 5. NORMALISE  (scale all values to 0–1)
# =============================================================================
print("\nNormalising data...")
scaler       = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df[FEATURES])   # fit ONLY on train
val_scaled   = scaler.transform(val_df[FEATURES])
test_scaled  = scaler.transform(test_df[FEATURES])

# Save the scaler — we need it later to convert predictions back to MW
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("  Scaler saved to scaler.pkl")


# =============================================================================
# 6. CREATE SEQUENCES
# Each sample = (past 168 hours → predict next 24 hours)
# =============================================================================
print("\nCreating sequences...")

def make_sequences(data, target_idx, lookback, horizon):
    """Slide a window across the data to create (X, y) pairs."""
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback : i])            # past window
        y.append(data[i : i + horizon, target_idx]) # future target values
    return np.array(X), np.array(y)

target_idx = FEATURES.index(TARGET_COL)

X_train, y_train = make_sequences(train_scaled, target_idx, LOOKBACK, HORIZON)
X_val,   y_val   = make_sequences(val_scaled,   target_idx, LOOKBACK, HORIZON)
X_test,  y_test  = make_sequences(test_scaled,  target_idx, LOOKBACK, HORIZON)

print(f"  X_train shape: {X_train.shape}  → (samples, timesteps, features)")
print(f"  y_train shape: {y_train.shape}  → (samples, hours to predict)")

# Save sequences so lstm_models.py can load them directly
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy',   X_val)
np.save('y_val.npy',   y_val)
np.save('X_test.npy',  X_test)
np.save('y_test.npy',  y_test)


# =============================================================================
# DONE
# =============================================================================
print("\n" + "="*50)
print("Data preparation complete!")
print(f"  Training samples:   {len(X_train):,}")
print(f"  Validation samples: {len(X_val):,}")
print(f"  Test samples:       {len(X_test):,}")
print(f"  Lookback: {LOOKBACK}h  |  Horizon: {HORIZON}h")
print("="*50)
print("\nNow run:  python lstm_models.py")