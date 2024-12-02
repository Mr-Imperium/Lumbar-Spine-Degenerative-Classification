# Lumbar Spine MRI Analysis

A Streamlit web application for analyzing lumbar spine MRI scans.

## Features
- Upload DICOM images
- Analyze three conditions:
  - Spinal Canal Stenosis
  - Neural Foraminal Narrowing
  - Subarticular Stenosis
- Get severity predictions
- Download analysis results

## Usage
1. Upload a DICOM (.dcm) file
2. Click "Analyze"
3. View results
4. Download CSV report

## Technical Details
- Built with PyTorch and Streamlit
- Uses ResNet18 backbone
- Handles DICOM medical imaging format
