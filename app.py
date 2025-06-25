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

# Grad-CAM
@st.cache_resource
def get_gradcam():
    return GradCAM(model, target_layer="layer4")

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

cam_extractor = get_gradcam()

# UI
st.title("🩺 COVID-19 Chest X-ray Classifier with Grad-CAM")
st.write("Upload a chest X-ray image to classify it and visualize model attention.")

uploaded_file = st.file_uploader("📤 Upload a chest X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Clear old hooks and re-enable
    cam_extractor.clear_hooks()
    cam_extractor._hooks_enabled = True

    # Forward pass with gradient tracking
    input_tensor.requires_grad = True
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    score = output[0, pred_class]

    # Backward pass to compute gradients
    model.zero_grad()
    score.backward()

    # Generate CAM (now gradients are available)
    cams = cam_extractor(pred_class, output)
    cam = cams[0]

    # Resize CAM to match input image
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu()

    # Prediction display
    probabilities = F.softmax(output[0].detach(), dim=0)
    st.markdown(f"### 🧠 Prediction: **{CLASSES[pred_class]}**")
    st.markdown("#### 🔍 Class Probabilities:")
    for i, prob in enumerate(probabilities):
        st.write(f"- {CLASSES[i]}: {prob.item():.4f}")

    # Grad-CAM overlay
    original = to_pil_image(transform(image))
    fig, ax = plt.subplots()
    ax.imshow(original)
    ax.imshow(cam, cmap="jet", alpha=0.5)
    ax.axis("off")
    st.markdown("### 🔥 Grad-CAM Heatmap")
    st.pyplot(fig)
