import streamlit as st
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image

# Load the pre-trained Dino model
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

model.head = nn.Identity()

# Set up image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_ = model.eval()

# Streamlit app
def main():
    st.title("Dino Embeddings Generator")

    # Upload and display the input image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Generate embeddings on button click
        if st.button("Generate Embeddings"):
            image_tensor = transform(image.convert("RGB")) 
            batch_tensor = image_tensor.unsqueeze(0)
            embeddings = model(batch_tensor)
            st.write(embeddings)

if __name__ == "__main__":
    main()
