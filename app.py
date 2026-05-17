import streamlit as st
import requests
import base64
from PIL import Image
import io
import os

# --- API Configuration ---
# Set a default placeholder, but prioritize the environment variable if available
API_URL = os.environ.get("API_URL", "https://your-space-name.hf.space/upscale")

st.set_page_config(page_title="Image Rescaling Tool", page_icon="🖼️", layout="centered")

# --- UI Headers ---
st.title("🖼️ AI Image Rescaler")
st.write(
    "Upload an image and specify a scaling factor. "
    "The heavy lifting is done by an external FastAPI backend."
)

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose an image to upscale...", 
    type=["png", "jpg", "jpeg"]
)

# --- Rescale Parameters ---
# The backend expects a scale_factor (integer from 1 to 16) rather than arbitrary w/h
scale_factor = st.slider("Scaling Factor (x)", min_value=2, max_value=8, value=4)
algorithm = st.selectbox("Interpolation Algorithm", ["bilinear", "nearest"])

# --- Submit Button & Processing ---
if st.button("Rescale Image"):
    if uploaded_file is None:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Processing image through the backend API..."):
            try:
                # Prepare the file for the multipart/form-data POST request
                files = {"image": (uploaded_file.name, uploaded_file.getvalue(), "image/png")}
                data = {
                    "scale_factor": scale_factor,
                    "algorithm": algorithm
                }
                
                # Send the POST request to the API
                response = requests.post(API_URL, files=files, data=data, timeout=60)
                
                if response.status_code == 200:
                    result_data = response.json()
                    
                    # The FastAPI backend returns the image as a base64 encoded string
                    image_b64 = result_data.get("image_b64")
                    elapsed = result_data.get("elapsed", 0)
                    
                    if image_b64:
                        # Decode the base64 string back into bytes
                        img_bytes = base64.b64decode(image_b64)
                        rescaled_image = Image.open(io.BytesIO(img_bytes))
                        
                        st.success(f"Successfully rescaled in {elapsed} seconds!")
                        
                        # Display the images side-by-side
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Original Image")
                            st.image(uploaded_file, use_column_width=True)
                        with col2:
                            st.subheader("Rescaled Image")
                            st.image(rescaled_image, use_column_width=True)
                            
                        # Download Button
                        st.download_button(
                            label="Download Rescaled Image",
                            data=img_bytes,
                            file_name=f"rescaled_{uploaded_file.name}",
                            mime="image/png"
                        )
                    else:
                        st.error("API response didn't contain an image payload.")
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the backend API: {e}")
