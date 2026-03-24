"""
Streamlit demo app - Upload an image and get top-5 food predictions.
"""
import streamlit as st
import numpy as np
from PIL import Image


def main():
    st.set_page_config(page_title="Food Classifier", page_icon="🍕", layout="wide")
    st.title("🍕 Food Image Classifier")
    st.markdown("Upload a food image to get predictions from our PRML models!")

    uploaded = st.file_uploader("Choose a food image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", width=400)

        st.info("Model prediction would go here after training.")
        st.markdown("**To use:**")
        st.markdown("1. Train models using `scripts/run_experiment.py`")
        st.markdown("2. Save the best model")
        st.markdown("3. Load it here for inference")


if __name__ == "__main__":
    main()
