"""
Streamlit Web App for Energy Demand Forecasting
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import models
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Energy Demand Forecasting",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">⚡ Energy Demand Forecasting System</p>', unsafe_allow_html=True)
st.markdown("### 24-Hour Ahead Prediction using LSTM Models")

# Load data and scaler (cached for performance)
@st.cache_data
def load_data():
    """Load the featured dataset"""
    df = pd.read_csv('pjm_featured.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df

@st.cache_resource
def load_scaler():
    """Load the scaler"""
    with open('scaler.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_model(model_name):
    """Load a trained model"""
    return models.load_model(f'saved_models/{model_name}.keras')

# Load resources
try:
    df = load_data()
    scaler = load_scaler()
    st.success("✓ Data and scaler loaded successfully")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar - Model selection and settings
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["LSTM", "BiLSTM", "SimpleRNN", "EncoderDecoder"],
    help="Choose which trained model to use for prediction"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About the Models")
st.sidebar.markdown("""
- **LSTM**: Best overall performance
- **BiLSTM**: Reads data forwards & backwards
- **SimpleRNN**: Baseline model
- **EncoderDecoder**: Good for longer horizons
""")

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📅 Select Prediction Date")
    
    # Get test period range
    test_start = pd.Timestamp('2017-03-24')
    test_end = pd.Timestamp('2018-08-03')
    
    st.info(f"Available date range:\n\n{test_start.date()} to {test_end.date()}")
    
    # Date input
    selected_date = st.date_input(
        "Pick a date",
        value=datetime(2017, 5, 15),
        min_value=test_start.date(),
        max_value=test_end.date()
    )
    
    # Time input
    selected_time = st.time_input(
        "Pick a time",
        value=datetime.strptime("14:00", "%H:%M").time()
    )
    
    # Combine date and time
    selected_datetime = pd.Timestamp(datetime.combine(selected_date, selected_time))
    
    st.markdown(f"**Selected timestamp:**  \n`{selected_datetime}`")
    
    # Predict button
    predict_button = st.button("🔮 Generate Forecast", type="primary", use_container_width=True)

with col2:
    st.subheader("📈 Forecast Results")
    
    if predict_button:
        try:
            with st.spinner(f"Loading {model_choice} model and generating forecast..."):
                # Load model
                model = load_model(model_choice)
                
                # Find timestamp in dataset
                target_idx = df[df['Datetime'] == selected_datetime].index[0]
                
                # Check bounds
                if target_idx < 168:
                    st.error("❌ Need at least 168 hours before this timestamp. Pick a later date.")
                    st.stop()
                
                if target_idx + 24 > len(df):
                    st.error("❌ Need 24 hours after this timestamp. Pick an earlier date.")
                    st.stop()
                
                # Get input window (168 hours)
                input_start_idx = target_idx - 168
                input_window = df.iloc[input_start_idx:target_idx]
                
                # Get actual values (24 hours)
                actual_window = df.iloc[target_idx:target_idx + 24]
                
                # Feature columns
                feature_cols = ['PJME', 'PJMW', 'DAYTON', 'AEP', 'DUQ', 
                               'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 
                               'IsWeekend', 'DayOfWeek']
                
                # Normalize and reshape
                input_scaled = scaler.transform(input_window[feature_cols].values)
                input_scaled = input_scaled.reshape(1, 168, 11)
                
                # Predict
                pred_scaled = model.predict(input_scaled, verbose=0)
                
                # Inverse transform
                n_feat = scaler.n_features_in_
                dummy = np.zeros((24, n_feat))
                dummy[:, 0] = pred_scaled[0]
                pred_pjme = scaler.inverse_transform(dummy)[:, 0]
                
                # Get actual PJME values
                actual_pjme = actual_window['PJME'].values
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(actual_pjme, pred_pjme))
                mae = mean_absolute_error(actual_pjme, pred_pjme)
                mape = mean_absolute_percentage_error(actual_pjme, pred_pjme) * 100
                
            # Display metrics
            st.success("✅ Forecast generated successfully!")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("RMSE", f"{rmse:,.0f} MW")
            metric_col2.metric("MAE", f"{mae:,.0f} MW")
            metric_col3.metric("MAPE", f"{mape:.2f}%")
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Hour': range(1, 25),
                'Predicted': pred_pjme,
                'Actual': actual_pjme,
                'Error': pred_pjme - actual_pjme
            })
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(results_df['Hour'], results_df['Predicted'], 
                   marker='o', linewidth=2.5, label='Predicted', color='#ff7f0e')
            ax.plot(results_df['Hour'], results_df['Actual'], 
                   marker='s', linewidth=2.5, label='Actual', color='#1f77b4', alpha=0.7)
            ax.fill_between(results_df['Hour'], 
                            results_df['Predicted'], 
                            results_df['Actual'], 
                            alpha=0.2, color='gray')
            ax.set_title(f'24-Hour PJME Energy Demand Forecast\nModel: {model_choice} | Starting: {selected_datetime}', 
                        fontweight='bold', fontsize=13)
            ax.set_xlabel('Hour Ahead', fontsize=11)
            ax.set_ylabel('Energy Demand (MW)', fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Show data table
            with st.expander("📋 View Detailed Results"):
                st.dataframe(
                    results_df.style.format({
                        'Predicted': '{:.1f}',
                        'Actual': '{:.1f}',
                        'Error': '{:.1f}'
                    }),
                    use_container_width=True
                )
            
            # Download option
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"forecast_{selected_datetime.strftime('%Y%m%d_%H%M')}_{model_choice}.csv",
                mime="text/csv"
            )
            
        except IndexError:
            st.error(f"❌ Timestamp {selected_datetime} not found in dataset.")
        except Exception as e:
            st.error(f"❌ Error generating forecast: {e}")
    else:
        st.info("👆 Select a date and time, then click 'Generate Forecast'")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🎓 PJM Energy Demand Forecasting System | LSTM Models</p>
    <p>Data period: 2005-2018 | Test period: 2017-2018</p>
</div>
""", unsafe_allow_html=True)
