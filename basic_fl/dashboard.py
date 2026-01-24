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

# --- Sidebar Navigation ---
st.sidebar.title("🏥 Federated Learning")
page = st.sidebar.radio("Go to", ["Training Monitor", "Model Inference"])

if page == "Training Monitor":
    st.title("📊 Training Monitor")
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
        if os.path.exists(config.METRICS_FILE):
            try:
                df = pd.read_csv(config.METRICS_FILE)
                
                if not df.empty and os.path.getmtime(config.METRICS_FILE) != last_update:
                    last_update = os.path.getmtime(config.METRICS_FILE)
                    
                    # Update Charts
                    chart_acc.line_chart(df.set_index("round")["accuracy"])
                    chart_loss.line_chart(df.set_index("round")["loss"])
                    
                    # Show Data Table
                    with st.expander("Raw Data Logs"):
                        st.dataframe(df)
                        
            except Exception as e:
                st.warning(f"Reading data... {e}")
                
        time.sleep(2) # Refresh every 2 seconds

elif page == "Model Inference":
    st.title("🔍 Model Inference")
    st.markdown("Upload an image to test the latest global model.")

    if not os.path.exists(config.MODEL_PATH):
        st.warning("No global model found. Please wait for at least one training round to complete.")
    else:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('L') # Convert to Grayscale
            st.image(image, caption='Uploaded Image', use_column_width=False, width=150)
            
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
                model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device(config.DEVICE)))
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
                st.error(f"Error loading model or running inference: {e}")
