import streamlit as st
import pandas as pd
import time
import os

CSV_FILE = "fl_metrics.csv"

st.set_page_config(page_title="FL Training Monitor", layout="wide")
st.title("🏥 Federated Learning Training Dashboard")
st.markdown("Monitoring training across Devices.")

# Placeholders for charts
col1, col2 = st.columns(2)
with col1:
    st.subheader("Global Accuracy")
    chart_acc = st.empty()
with col2:
    st.subheader("Global Loss")
    chart_loss = st.empty()

st.info("Waiting for training rounds to complete...")

last_update = 0

while True:
    if os.path.exists(CSV_FILE):
        # Read CSV
        try:
            df = pd.read_csv(CSV_FILE)
            
            if not df.empty and os.path.getmtime(CSV_FILE) != last_update:
                last_update = os.path.getmtime(CSV_FILE)
                
                # Update Charts
                chart_acc.line_chart(df.set_index("round")["accuracy"])
                chart_loss.line_chart(df.set_index("round")["loss"])
                
                # Show Data Table
                with st.expander("Raw Data Logs"):
                    st.dataframe(df)
                    
        except Exception as e:
            st.warning(f"Reading data... {e}")
            
    time.sleep(2) # Refresh every 2 seconds