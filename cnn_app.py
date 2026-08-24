import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

st.title("Fashion CNN Classifier")
st.write("Upload a clothing item image to classify.")

@st.cache_resource
def load_industry_model():
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.eval()
    return model

try:
    model = load_industry_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

clothing_categories = {
    0: "T-shirt 👕", 1: "Trouser 👖", 2: "Pullover 🧥", 3: "Dress 👗", 4: "Coat 🧥",
    5: "Sandal 👡", 6: "Shirt 👔", 7: "Sneaker 👟", 8: "Bag 👜", 9: "Ankle boot 🥾"
}


imagenet_to_fashion = {
    610: 0, # jersey -> T-shirt
    843: 1, # jean -> Trouser
    472: 2, # cardigan -> Pullover
    543: 3, # gown -> Dress
    412: 4, # trench coat -> Coat
    771: 5, # sandal -> Sandal
    617: 6, # lab coat -> Shirt
    802: 7, # running shoe -> Sneaker
    414: 8, # backpack -> Bag
    415: 9  # ankle boot -> Ankle boot
}

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1).squeeze(0)
        
       
        best_fashion_idx = 0
        max_prob = -1.0
        
        for imgnet_idx, fashion_idx in imagenet_to_fashion.items():
            if probabilities[imgnet_idx].item() > max_prob:
                max_prob = probabilities[imgnet_idx].item()
                best_fashion_idx = fashion_idx
                
    st.success(f"🎉 Model's Predicted Guess: **{clothing_categories[best_fashion_idx]}**")
