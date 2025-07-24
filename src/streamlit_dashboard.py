import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Cấu hình trang
st.set_page_config(
    page_title="Real-time Video Classification Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Đường dẫn đến file dữ liệu
DATA_PATH = "/home/anhkhoa/spark_video_streaming/checkpoint/result_files/latest_predictions.parquet"

def load_data():
    """Load data from parquet file"""
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_parquet(DATA_PATH)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def calculate_metrics(df):
    """Calculate accuracy and other metrics"""
    if df.empty:
        return {}
    
    accuracy = accuracy_score(df['label'], df['predicted_label'])
    
    # Count predictions by label
    label_counts = df['predicted_label'].value_counts()
    
    # Confusion matrix
    cm = confusion_matrix(df['label'], df['predicted_label'])
    
    return {
        'accuracy': accuracy,
        'label_counts': label_counts,
        'confusion_matrix': cm,
        'total_predictions': len(df)
    }

def create_label_distribution_chart(label_counts):
    """Create bar chart for label distribution"""
    fig = px.bar(
        x=label_counts.index,
        y=label_counts.values,
        labels={'x': 'Predicted Labels', 'y': 'Count'},
        title='Distribution of Predicted Labels',
        color=label_counts.values,
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        height=400,
        showlegend=False
    )
    return fig

def create_accuracy_over_time_chart(df):
    """Create line chart showing accuracy over time"""
    if df.empty:
        return go.Figure()
    
    # Calculate rolling accuracy
    df_sorted = df.sort_values('timestamp')
    window_size = min(50, len(df_sorted))
    
    rolling_accuracy = []
    timestamps = []
    
    for i in range(window_size, len(df_sorted) + 1):
        window_df = df_sorted.iloc[i-window_size:i]
        acc = accuracy_score(window_df['label'], window_df['predicted_label'])
        rolling_accuracy.append(acc)
        timestamps.append(window_df['timestamp'].iloc[-1])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=rolling_accuracy,
        mode='lines+markers',
        name='Rolling Accuracy',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=f'Rolling Accuracy Over Time (Window: {window_size})',
        xaxis_title='Time',
        yaxis_title='Accuracy',
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

def create_confusion_matrix_heatmap(cm, labels):
    """Create confusion matrix heatmap"""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        text_auto=True,
        title='Confusion Matrix'
    )
    fig.update_layout(height=500)
    return fig

def create_confidence_distribution(df):
    """Create histogram of prediction confidence"""
    if df.empty or 'prediction_confidence' not in df.columns:
        return go.Figure()
    
    fig = px.histogram(
        df,
        x='prediction_confidence',
        nbins=20,
        title='Distribution of Prediction Confidence',
        labels={'prediction_confidence': 'Confidence Score', 'count': 'Frequency'}
    )
    fig.update_layout(height=400)
    return fig

# Main dashboard
def main():
    st.title("🎬 Real-time Video Classification Dashboard")
    st.markdown("Dashboard for monitoring video classification model predictions in real-time")
    
    # Load data first
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 30, 5)
    
    # Display refresh status
    if auto_refresh:
        st.sidebar.success(f"🔄 Auto-refresh: ON ({refresh_interval}s)")
    else:
        st.sidebar.info("⏸️ Auto-refresh: OFF")
    
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Show last update time (only if data exists)
    if not df.empty:
        last_update = df['timestamp'].max()
        st.sidebar.info(f"📅 Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.sidebar.info("📅 No data loaded yet")
    
    # Data source info
    st.sidebar.markdown("---")
    st.sidebar.markdown("📂 **Data Source**")
    st.sidebar.code(DATA_PATH, language=None)
    # Check if data is empty and handle accordingly
    if df.empty:
        st.warning("⚠️ No data available yet. The dashboard is waiting for predictions...")
        st.info(f"Expected data path: {DATA_PATH}")
        st.info("💡 Make sure your streaming process is running and generating predictions.")
        
        # Show a placeholder while waiting
        if auto_refresh:
            st.info(f"⏱️ Auto-refreshing every {refresh_interval} seconds...")
            
            # Countdown with progress bar
            placeholder = st.empty()
            progress_bar = st.progress(0)
            
            for i in range(refresh_interval):
                remaining = refresh_interval - i
                placeholder.info(f"⏳ Checking for new data in {remaining} seconds...")
                progress_bar.progress((i + 1) / refresh_interval)
                time.sleep(1)
            
            placeholder.empty()
            progress_bar.empty()
            st.rerun()
        else:
            st.info("🔄 Enable 'Auto Refresh' in the sidebar to automatically check for new data.")
        return
    
    # Data is available - proceed with dashboard
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", metrics['total_predictions'])
    
    with col2:
        st.metric("Overall Accuracy", f"{metrics['accuracy']:.3f}")
    
    with col3:
        st.metric("Unique Labels", len(metrics['label_counts']))
    
    with col4:
        if df.empty:
            st.metric("Latest Prediction", "No data")
        else:
            latest_time = df['timestamp'].max()
            time_diff = datetime.now() - latest_time.replace(tzinfo=None)
            if time_diff.total_seconds() < 60:
                st.metric("Latest Prediction", f"{int(time_diff.total_seconds())}s ago", delta="🟢 Active")
            else:
                st.metric("Latest Prediction", latest_time.strftime("%H:%M:%S"), delta="🔴 Stale")
    
    # Main charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            create_label_distribution_chart(metrics['label_counts']),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            create_accuracy_over_time_chart(df),
            use_container_width=True
        )
    
    # Additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        labels = sorted(df['label'].unique())
        fig_cm = create_confusion_matrix_heatmap(metrics['confusion_matrix'], labels)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        # Confidence Distribution
        fig_conf = create_confidence_distribution(df)
        st.plotly_chart(fig_conf, use_container_width=True)
    
    # Recent predictions table
    st.subheader("📋 Recent Predictions")
    recent_df = df.nlargest(20, 'timestamp')[['id', 'label', 'predicted_label', 'prediction_confidence', 'timestamp']]
    recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Color code correct/incorrect predictions
    def highlight_accuracy(row):
        if row['label'] == row['predicted_label']:
            return ['background-color: #d4edda'] * len(row)
        else:
            return ['background-color: #f8d7da'] * len(row)
    
    st.dataframe(
        recent_df.style.apply(highlight_accuracy, axis=1),
        use_container_width=True
    )
    
    # Detailed statistics
    with st.expander("📊 Detailed Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Label Distribution")
            st.dataframe(metrics['label_counts'].to_frame('Count'))
        
        with col2:
            st.subheader("Accuracy by Label")
            label_accuracy = []
            for label in df['label'].unique():
                label_df = df[df['label'] == label]
                if not label_df.empty:
                    acc = accuracy_score(label_df['label'], label_df['predicted_label'])
                    label_accuracy.append({'Label': label, 'Accuracy': acc, 'Count': len(label_df)})
            
            if label_accuracy:
                acc_df = pd.DataFrame(label_accuracy)
                st.dataframe(acc_df.round(3))
    
    # Auto refresh logic (moved to bottom for better UX)
    if auto_refresh:
        # Use st.empty() for better refresh experience
        placeholder = st.empty()
        
        # Countdown timer
        for remaining in range(refresh_interval, 0, -1):
            placeholder.info(f"⏱️ Next refresh in {remaining} seconds...")
            time.sleep(1)
        
        placeholder.empty()
        st.rerun()

if __name__ == "__main__":
    main()