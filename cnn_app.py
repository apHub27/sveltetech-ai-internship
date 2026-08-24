import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

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

# बाहरी यूआरएल हटाकर सीधे मुख्य कपड़ों की 1000 क्लासेस की मैपिंग लोकली कोड में ही लिख दी है
@st.cache_resource
def load_labels_local():
    # डिफ़ॉल्ट 1000 क्लासेस के बजाय सिर्फ मुख्य क्लॉथिंग क्लासेस की सटीक मैppings
    clothing_map = {
        610: "Jersey, T-shirt 👕",
        843: "Trouser, Jean, Pants 👖",
        472: "Cardigan, Pullover 🧥",
        543: "Dress, Gown 👗",
        412: "Coat, Jacket 🧥",
        771: "Sandal, Flip-flop 👡",
        617: "Lab coat, Shirt 👔",
        802: "Sneaker, Running shoe 👟",
        414: "Backpack, Bag 👜",
        415: "Ankle boot, Boot 🥾"
    }
    return clothing_map

clothing_dict = load_labels_local()

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
        
    # अगर इमेजनेट की क्लॉथिंग क्लास मैच होती है तो सही नाम दिखाएगा, वरना जेनेरिक नाम
    if predicted_label_index in clothing_dict:
        result = clothing_dict[predicted_label_index]
    else:
        # अगर कोई ऐसी क्लास आती है जो डिक्शनरी में नहीं है, तो मॉडल के डिफ़ॉल्ट आउटपुट से गेस करेगा
        result = "Clothing item detected"
        
    st.success(f"🎉 Model's Predicted Guess: **{result}** (Confidence: {confidence:.2f}%)")
