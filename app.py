import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torchvision.transforms.functional import to_pil_image

# Class names
CLASSES = ["COVID", "Normal", "Viral Pneumonia"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load("covid_classifier.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Initialize Grad-CAM
@st.cache_resource
def get_gradcam():
    return GradCAM(model, target_layer="layer4")

cam_extractor = get_gradcam()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# UI
st.title("🩺 COVID-19 Chest X-ray Classifier with Grad-CAM")
st.write("Upload a chest X-ray image to classify it and visualize model attention.")

uploaded_file = st.file_uploader("📤 Upload a chest X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(DEVICE).requires_grad_()

    # Clear Grad-CAM hooks
    cam_extractor.clear_hooks()

    # Forward & backward pass for Grad-CAM
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    score = output[0, pred_class]
    model.zero_grad()
    score.backward(retain_graph=True)

    # Extract CAM
    cam = cam_extractor(pred_class)[0]
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu()

    # Display prediction
    probabilities = F.softmax(output[0].detach(), dim=0)
    st.markdown(f"### 🧠 Prediction: **{CLASSES[pred_class]}**")
    st.markdown("#### 🔍 Class Probabilities:")
    for i, prob in enumerate(probabilities):
        st.write(f"- {CLASSES[i]}: {prob.item():.4f}")

    # Grad-CAM overlay
    image_tensor = transform(image)
    original = to_pil_image(image_tensor)
    fig, ax = plt.subplots()
    ax.imshow(original)
    ax.imshow(cam, cmap="jet", alpha=0.5)
    ax.axis("off")
    st.markdown("### 🔥 Grad-CAM Heatmap")
    st.pyplot(fig)
