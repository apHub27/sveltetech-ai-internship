import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

st.title("Fashion CNN Classifier")

@st.cache_resource
def load_full_model():
    return torch.load('full_fashion_model.pth', map_location=torch.device('cpu'))

try:
    model = load_full_model()
    model.eval()
except Exception as e:
    st.error(f"Error: {e}")

clothing_categories = {
    0: "T-shirt 👕", 1: "Trouser 👖", 2: "Pullover 🧥", 3: "Dress 👑", 4: "Coat 🧥",
    5: "Sandal 👡", 6: "Shirt 👔", 7: "Sneaker 👟", 8: "Bag 👜", 9: "Ankle boot 🥾"
}

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    input_tensor = transform(image).unsqueeze(0).float()
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted_label_index = torch.argmax(output, dim=1).item()
        
    st.success(f"🎉 Model's Predicted Guess: {clothing_categories[predicted_label_index]}")
