# UI Sketch Classification Project

## Overview

This project focuses on classifying UI sketch components (e.g., buttons, cards, images) using machine learning and deep learning methods. The main steps involve data preprocessing, feature extraction, and model development, utilizing both traditional machine learning and deep learning architectures.

## Data Processing

- **Dataset Preparation**: The dataset contains labeled categories (button, card, image) from UI sketches.
- **Image Resizing**: Each image is resized to a standard 150x150 pixel format for uniformity in training.
- **Data Splitting**: The data is split into training and testing sets, with further preprocessing applied to optimize model input.

## Feature Extraction

To extract meaningful features from images:
- **Grey-Level Co-Occurrence Matrix (GLCM)**: Various GLCM properties like energy, correlation, dissimilarity, homogeneity, and contrast are computed at multiple orientations and distances, providing a robust feature set to describe texture and spatial relationships in the images.
- **Feature Selection**: SelectKBest is used to choose the top features, optimizing the dataset for high accuracy and low computational load.

## Model Development

### Machine Learning Models

1. **LightGBM**:
   - Parameters: Dart boosting type, multi-class objective, learning rate, maximum depth, and leaf nodes.
   - **Performance**: Trained on the selected features, achieving baseline accuracy on test data.

2. **XGBoost**:
   - Trained with the same selected features, with model predictions evaluated for accuracy.
   - **Performance**: Results serve as a comparative baseline for subsequent deep learning models.

### Deep Learning Models

1. **Convolutional Neural Network (CNN)**:
   - **Architecture**: Sequential model with convolutional layers (32, 64 filters) followed by max pooling and dense layers.
   - **Training**: The CNN is trained with augmented data using an ImageDataGenerator for robust generalization.

2. **Transfer Learning with ResNet and VGG**:
   - **ResNet and VGG**: Pretrained models adapted for UI classification with fine-tuning on the final layers.
   - **Data Augmentation**: Enhanced with FastAI transforms, including rotation, padding, and normalization.

## Evaluation

The models are evaluated using:
- **Accuracy**: To measure overall model performance.
- **Confusion Matrix**: For insight into class-specific performance.
- **Classification Report**: Detailing precision, recall, and F1 scores for each class.
  
### Visualization

- **Training and Validation Plots**: Loss and accuracy plots across epochs to monitor model learning.
- **Confusion Matrix Heatmaps**: For clear visualization of class prediction accuracy.

## Conclusion

The project demonstrates effective UI component classification using machine learning and deep learning approaches. CNN-based architectures, especially when enhanced with transfer learning, show promise in accurately identifying UI components from sketch data. Future directions could involve exploring additional architectures and expanding the dataset to improve generalization and robustness for diverse UI sketch datasets.
