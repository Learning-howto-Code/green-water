# if-water-cnn

CNN model to detect if water is present in toilet bowl images.

## Overview

This project uses a convolutional neural network to classify images as either containing water or not containing water in a toilet bowl.

## Training Improvements

### Overfitting Fix (Dec 2024)

**Problem**: The model was achieving 100% training accuracy but only 50% test accuracy, indicating severe overfitting.

**Solution**: Implemented the following changes to prevent overfitting:

1. **Enabled Data Augmentation**: Uncommented the data augmentation layer in the model, which applies random transformations (zoom, translation, flip) to training images to increase model generalization.

2. **Added Dropout Regularization**: Added a Dropout(0.5) layer before the final output layer to randomly drop 50% of neurons during training, preventing the model from relying too heavily on specific features.

3. **Enabled Early Stopping**: Added early stopping callback to halt training if validation loss doesn't improve for 3 consecutive epochs, preventing the model from overfitting to the training data.

These changes should significantly improve test accuracy by helping the model generalize better to unseen data.

## Files

- `cnn.py` - Main training script
- `test.py` - Test the trained model on test data
- `main.py` - Capture images from Raspberry Pi camera
- `split.py` - Split training data into train/validation sets
- `plot.py` - Visualize prediction results from logs
