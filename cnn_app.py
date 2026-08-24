import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json

st.title("Industry Standard Clothing Classifier")
st.write("Upload any real-world clothing image to classify.")

@st.cache_resource
def load_industry_model():
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.eval()
    return model

try:
    model = load_industry_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# पूरी 1000 क्लासेस की मैपिंग लोकली बिना किसी इंटरनेट लिंक के फिक्स कर दी है
@st.cache_resource
def load_labels_local():
    # क्लॉथिंग से जुड़े सभी 1000 इंडेक्स की मुख्य क्लासेस की मैपिंग
    import torchvision.models as models
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    return weights.meta["categories"]

categories = load_labels_local()

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
        predicted_label_index = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_label_index].item() * 100
        
    result = categories[predicted_label_index]
    st.success(f"🎉 Model's Predicted Guess: **{result.replace('_', ' ').capitalize()}** (Confidence: {confidence:.2f}%)")
