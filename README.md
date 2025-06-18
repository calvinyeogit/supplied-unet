# Supplied U-Net

> ⚠️ **WARNING: Known Issue** - The model currently produces zero-value predictions. See [Known Issues](#-known-issues-and-current-status) section for details and ongoing troubleshooting efforts.

A deep learning project implementing U-Net architecture for image segmentation, specifically designed for processing and analyzing microscopy images.

## ⚠️ Known Issues and Current Status

### Zero-Value Prediction Problem
The model is currently experiencing a critical issue where all predicted masks contain only zero-value pixels. This means the segmentation is failing to identify any features in the input images.

**Current Observations:**
- All predicted masks are completely black (pixel value = 0)
- This occurs consistently across all test images
- The issue persists despite successful training completion

**Troubleshooting Steps Taken:**
1. Verified training data:
   - Confirmed proper loading of image-mask pairs
   - Validated that training masks contain proper binary values (0 and 255)
   - Implemented visualization tools to verify data loading

2. Model Training:
   - Adjusted training epochs (tested with both 5 and 30 epochs)
   - Monitored loss values during training
   - Verified model architecture and forward pass

3. Prediction Pipeline:
   - Added threshold adjustment (currently set to 0.3)
   - Implemented prediction value analysis
   - Added detailed logging and visualization

**Next Steps:**
- Further investigation of model output values before thresholding
- Analysis of model weights and activation patterns
- Potential adjustments to loss function and optimization parameters

## Project Structure

```
supplied-unet/
├── dataset_wangwei/           # Main data directory
│   ├── forpredict/           # Test images for prediction
│   ├── images/               # Training images
│   ├── masks/                # Training masks/ground truth
│   └── predicted/            # Generated predictions
├── figures/                  # Generated plots and visualizations
├── UNetModela.py            # U-Net model architecture
├── Unet_Config.py           # Model configuration
├── dataset.py               # Dataset loading and preprocessing
├── Json_Functions.py        # JSON utility functions
├── Losses.py               # Loss functions
├── testUnetload.py         # Main training and prediction script
└── requirements.txt        # Python dependencies
```

## Description

This project implements a U-Net model for image segmentation tasks, particularly focused on microscopy image analysis. The implementation includes training, prediction, and visualization capabilities.

### Key Components

- **Model Architecture**: Implemented in `UNetModela.py`, based on the U-Net architecture
- **Dataset Handling**: Custom dataset loader in `dataset.py` for processing image-mask pairs
- **Training & Prediction**: Main functionality in `testUnetload.py`
- **Visualization**: Includes tools for displaying and saving prediction results

## Data Organization

- **Training Data**:
  - Images: `dataset_wangwei/images/`
  - Masks: `dataset_wangwei/masks/`
  - Format: TIFF images (.tif)

- **Prediction Data**:
  - Input: `dataset_wangwei/forpredict/`
  - Output: `dataset_wangwei/predicted/`
  - Format: Input as TIFF (.tif), predictions saved as JPEG (.jpg)

- **Visualizations**:
  - Stored in: `figures/`
  - Format: PNG files with timestamps
  - Naming: `prediction_comparison_YYYY-MM-DD_HHMMSS.png`

## Usage

1. **Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Training**:
   ```python
   python testUnetload.py
   ```
   - Default configuration: 5 epochs for testing, 30 for production
   - Model weights saved as `unet_modeld.pth`

3. **Prediction**:
   - Place input images in `dataset_wangwei/forpredict/`
   - Run the script to generate predictions
   - Results saved in `dataset_wangwei/predicted/`
   - Visualization plots saved in `figures/`

## Model Configuration

Key parameters in `Unet_Config.py`:
- Input channels: 1 (grayscale images)
- Output channels: 1 (binary segmentation)
- Base features: 16

## Visualization

The project includes comprehensive visualization tools:
- Side-by-side comparison of original images and predictions
- Grid layout display of multiple image pairs
- Automatic saving of visualization plots with timestamps
- Analysis of prediction pixel values

## Dependencies

Main dependencies include:
- PyTorch
- NumPy
- Pillow (PIL)
- Matplotlib

See `requirements.txt` for complete list and versions.

## Notes

- The model is configured for grayscale images
- Predictions are binarized with a threshold of 0.3
- Visualization plots are saved at 300 DPI for high quality
- Generated files (predictions, plots) are automatically organized in appropriate directories

## ⚠️ Known Issues and Current Status

### Zero-Value Prediction Problem
The model is currently experiencing a critical issue where all predicted masks contain only zero-value pixels. This means the segmentation is failing to identify any features in the input images.

**Current Observations:**
- All predicted masks are completely black (pixel value = 0)
- This occurs consistently across all test images
- The issue persists despite successful training completion

**Troubleshooting Steps Taken:**
1. Verified training data:
   - Confirmed proper loading of image-mask pairs
   - Validated that training masks contain proper binary values (0 and 255)
   - Implemented visualization tools to verify data loading

2. Model Training:
   - Adjusted training epochs (tested with both 5 and 30 epochs)
   - Monitored loss values during training
   - Verified model architecture and forward pass

3. Prediction Pipeline:
   - Added threshold adjustment (currently set to 0.3)
   - Implemented prediction value analysis
   - Added detailed logging and visualization

**Next Steps:**
- Further investigation of model output values before thresholding
- Analysis of model weights and activation patterns
- Potential adjustments to loss function and optimization parameters 