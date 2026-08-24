import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

st.title("Fashion CNN Classifier")
st.write("Upload a clothing item image to classify.")

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

@st.cache_resource
def load_model():
    model = FashionCNN()
    model.load_state_dict(torch.load('my_cnn_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

clothing_categories = {
    0: "T-shirt 👕", 1: "Trouser 👖", 2: "Pullover 🧥", 3: "Dress 👗", 4: "Coat 👑",
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
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted_label_index = torch.argmax(output).item()
        
    st.success(f"🎉 Model's Predicted Guess: {clothing_categories[predicted_label_index]}")
