import streamlit as st
import torch
import timm
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pydicom
import os
from PIL import Image
import io

# Model Definition
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

# Image Processing Functions
def process_dicom(dicom_file):
    try:
        # Read DICOM file
        dicom = pydicom.dcmread(dicom_file)
        image = dicom.pixel_array
        
        # Normalize image
        image = image.astype(float)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = (image * 255).astype(np.uint8)
        
        # Convert to 3 channels if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
            
        return image
    except Exception as e:
        st.error(f"Error processing DICOM file: {str(e)}")
        return None

def get_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

# Prediction Function
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MRIConditionModel(backbone='resnet18', num_conditions=3)
    
    # Load the trained model weights
    try:
        checkpoint = torch.load('best_mri_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, device

def predict(image, model, device):
    transform = get_transforms()
    
    # Transform image
    image_tensor = transform(image=image)['image']
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs)
    
    return probabilities.cpu().numpy()[0]

# Streamlit UI
def main():
    st.title("Lumbar Spine MRI Analysis")
    st.write("Upload a DICOM image for analysis of spinal conditions")
    
    # Load model
    model, device = load_model()
    if model is None:
        st.error("Failed to load model. Please check the model file.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a DICOM file", type=['dcm'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_file = "temp.dcm"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Process image
        image = process_dicom(temp_file)
        os.remove(temp_file)  # Clean up
        
        if image is not None:
            # Display the image
            st.image(image, caption='Uploaded MRI scan', use_column_width=True)
            
            # Make prediction
            if st.button('Analyze'):
                with st.spinner('Analyzing image...'):
                    predictions = predict(image, model, device)
                
                # Display results
                conditions = ['Spinal Canal Stenosis', 'Neural Foraminal Narrowing', 'Subarticular Stenosis']
                severities = ['Normal/Mild', 'Moderate', 'Severe']
                
                st.subheader("Analysis Results:")
                
                for i, condition in enumerate(conditions):
                    st.write(f"\n{condition}:")
                    for j, severity in enumerate(severities):
                        probability = predictions[i*3 + j]
                        prob_percentage = f"{probability*100:.1f}%"
                        st.progress(float(probability))
                        st.write(f"{severity}: {prob_percentage}")
                    st.write("---")
                
                # Download results
                results_df = pd.DataFrame({
                    'Condition': [cond for cond in conditions for _ in range(3)],
                    'Severity': severities * 3,
                    'Probability': [f"{p*100:.1f}%" for p in predictions]
                })
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="mri_analysis_results.csv",
                    mime="text/csv"
                )

if __name__ == '__main__':
    main()
