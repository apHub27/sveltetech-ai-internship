import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys

st.title("Fashion CNN Classifier")

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 14 * 14, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

sys.modules['__main__'].FashionCNN = FashionCNN

@st.cache_resource
def load_full_model():
    return torch.load('full_fashion_model.pth', map_location=torch.device('cpu'), weights_only=False)

try:
    model = load_full_model()
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")

clothing_categories = {
    0: "T-shirt 👕", 1: "Trouser 👖", 2: "Pullover 🧥", 3: "Dress 👗", 4: "Coat 🧥",
    5: "Sandal 👡", 6: "Shirt 👔", 7: "Sneaker 👟", 8: "Bag 👜", 9: "Ankle boot 🥾"
}

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    st.image(image, caption='Processed Image', use_container_width=True)
    
    # Mathematical Sync: Convert image into a standard 0.0 - 1.0 NumPy array first
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Cast directly to a clean PyTorch FloatTensor and add Batch/Channel dimensions
    input_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).float()
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted_label_index = torch.argmax(output, dim=1).item()
        
    st.success(f"🎉 Model's Predicted Guess: {clothing_categories[predicted_label_index]}")
