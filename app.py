import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
import pydicom
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set page config
st.set_page_config(
    page_title="Lumbar Spine Analysis",
    page_icon="🏥",
    layout="wide"
)

# Model definition (same as your original code)
class MRIConditionModel(nn.Module):
    def __init__(self, backbone='resnet18', num_conditions=3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            in_chans=3,
            features_only=False,
            num_classes=0
        )
        self.num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_conditions * 3)
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        x = self.classifier(features)
        return x

# Image preprocessing functions
def preprocess_image(image_array):
    # Create transform
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    # Ensure image is in correct format
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    
    # Apply transformations
    transformed = transform(image=image_array)
    return transformed['image'].unsqueeze(0)

def load_dicom(file):
    dicom = pydicom.dcmread(file)
    image = dicom.pixel_array
    image = image - image.min()
    image = image / (image.max() + 1e-8)
    image = (image * 255).astype(np.uint8)
    return image

def load_regular_image(file):
    image = Image.open(file)
    image = np.array(image.convert('RGB'))
    return image

# Prediction visualization functions
def create_prediction_plot(predictions):
    conditions = ['Spinal Canal Stenosis', 'Neural Foraminal Narrowing', 'Subarticular Stenosis']
    severities = ['Normal/Mild', 'Moderate', 'Severe']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    data = predictions.reshape(3, 3)
    
    sns.heatmap(data, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=severities, yticklabels=conditions,
                vmin=0, vmax=1)
    
    plt.title('Condition Severity Predictions')
    plt.tight_layout()
    
    return fig

def create_prediction_df(predictions):
    conditions = ['Spinal Canal Stenosis', 'Neural Foraminal Narrowing', 'Subarticular Stenosis']
    severities = ['Normal/Mild', 'Moderate', 'Severe']
    
    data = []
    pred_reshape = predictions.reshape(3, 3)
    
    for i, condition in enumerate(conditions):
        for j, severity in enumerate(severities):
            data.append({
                'Condition': condition,
                'Severity': severity,
                'Probability': pred_reshape[i, j]
            })
    
    return pd.DataFrame(data)

# Main Streamlit app
def main():
    st.title("Lumbar Spine MRI Analysis")
    st.write("Upload an image for analysis of spinal conditions")
    
    # Model loading
    @st.cache_resource
    def load_model():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MRIConditionModel(backbone='resnet18', num_conditions=3)
        checkpoint = torch.load('best_mri_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, device

    try:
        model, device = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # File upload
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg', 'dcm'])
    
    if uploaded_file is not None:
        try:
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            # Load and display image
            if uploaded_file.name.lower().endswith('.dcm'):
                image = load_dicom(uploaded_file)
            else:
                image = load_regular_image(uploaded_file)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess and predict
            preprocessed_image = preprocess_image(image)
            with torch.no_grad():
                predictions = torch.sigmoid(model(preprocessed_image.to(device))).cpu().numpy()[0]
            
            # Display results
            with col2:
                st.subheader("Analysis Results")
                fig = create_prediction_plot(predictions)
                st.pyplot(fig)
            
            # Create downloadable results
            results_df = create_prediction_df(predictions)
            
            # Download buttons
            st.subheader("Download Results")
            col3, col4 = st.columns(2)
            
            with col3:
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"spine_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col4:
                # Save plot to bytes buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    label="Download Plot",
                    data=buf,
                    file_name=f"spine_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
