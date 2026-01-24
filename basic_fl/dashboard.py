import streamlit as st
import pandas as pd
import time
import os
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import config
from utils import Net

st.set_page_config(page_title="FL Dashboard", layout="wide")

# --- Header ---
st.title("🏥 Federated Learning Dashboard")
st.markdown("Monitor training and test models across the federation.")

# --- Sidebar ---
st.sidebar.title("Settings")
auto_refresh = st.sidebar.checkbox("Auto-refresh Monitor", value=False, help="Refresh the dashboard every 5 seconds to see live updates.")

# --- Tabs ---
tab_monitor, tab_inference = st.tabs(["📊 Training Monitor", "🔍 Model Inference"])

with tab_monitor:
    st.subheader("Global Training Progress")
    
    if os.path.exists(config.GLOBAL_LOG_PATH):
        try:
            df_global = pd.read_csv(config.GLOBAL_LOG_PATH)
            if not df_global.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.line_chart(df_global.set_index("round")["accuracy"], use_container_width=True)
                    st.caption("Global Accuracy")
                with col2:
                    st.line_chart(df_global.set_index("round")["loss"], use_container_width=True)
                    st.caption("Global Loss")
                
                with st.expander("Raw Global Data"):
                    st.dataframe(df_global)
            else:
                st.info("Global log is empty. Waiting for training...")
        except Exception as e:
            st.error(f"Error reading global log: {e}")
    else:
        st.info("Waiting for global metrics...")

    st.divider()
    st.subheader("Client-wise Metrics")
    
    # Find all client logs
    client_logs = []
    if os.path.exists(config.LOGS_DIR):
        for f in os.listdir(config.LOGS_DIR):
            if f.startswith("client_") and f.endswith("_metrics.csv"):
                cid = f.replace("client_", "").replace("_metrics.csv", "")
                client_logs.append((cid, os.path.join(config.LOGS_DIR, f)))
    
    if client_logs:
        # Sort by Client ID
        client_logs.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
        
        # Display each client in an expander
        for cid, log_path in client_logs:
            with st.expander(f"Client {cid} Metrics", expanded=True):
                try:
                    df_client = pd.read_csv(log_path)
                    if not df_client.empty:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.line_chart(df_client.set_index("round")["accuracy"])
                        with c2:
                            st.line_chart(df_client.set_index("round")["loss"])
                    else:
                        st.write("No data yet for this client.")
                except Exception as e:
                    st.error(f"Error reading log for Client {cid}: {e}")
    else:
        st.info("No client metrics found yet.")

with tab_inference:
    st.subheader("Model Inference")
    st.markdown("Upload an image to test a model.")

    # Model Selection
    model_source = st.radio("Model Source", ["Global", "Client"], horizontal=True, key="inference_source")
    
    model_path = None
    if model_source == "Global":
        model_path = config.GLOBAL_MODEL_PATH
    else:
        client_models = []
        if os.path.exists(config.MODELS_DIR):
            for f in os.listdir(config.MODELS_DIR):
                if f.startswith("client_") and f.endswith(".pth"):
                    cid = f.replace("client_", "").replace(".pth", "")
                    client_models.append(cid)
        
        if client_models:
            selected_client = st.selectbox("Select Client Model", sorted(client_models, key=lambda x: int(x) if x.isdigit() else x))
            model_path = config.get_client_model_path(selected_client)
        else:
            st.warning("No client models found.")

    if model_path and not os.path.exists(model_path):
        st.warning(f"Selected model not found at {model_path}")
    elif model_path:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="inference_uploader")

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('L')
            st.image(image, caption='Uploaded Image', width=150)
            
            # Preprocessing
            transform = Compose([
                Resize((28, 28)),
                ToTensor(),
                Normalize(mean=[.5], std=[.5])
            ])
            img_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
            
            # Load Model
            model = Net()
            try:
                model.load_state_dict(torch.load(model_path, map_location=torch.device(config.DEVICE)))
                model.eval()
                
                with torch.no_grad():
                    output = model(img_tensor)
                    prediction = (output > 0.5).float().item()
                    confidence = output.item() if prediction == 1 else 1 - output.item()
                
                # Result Display
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.error(f"**Positive (Pneumonia Detected)**")
                else:
                    st.success(f"**Negative (Normal)**")
                
                st.write(f"Confidence: {confidence:.2%}")
                
            except Exception as e:
                st.error(f"Error running inference: {e}")

# --- Global Auto-refresh ---
if auto_refresh:
    time.sleep(5)
    st.rerun()
