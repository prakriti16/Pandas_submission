# Report

## Approach
We explored various popular image classification techniques to address our problem. Initially, we employed Convolutional Neural Networks (CNNs), achieving an accuracy of around 0.7. Upon analyzing the data distribution across different labels, we found it biased. We implemented several advanced techniques to improve performance, ultimately enhancing our AUC-ROC to 0.93 and accuracy to 0.85. These techniques included:

- K-fold cross-validation
- Data augmentation
- Class weights
- Focal Loss
- SMOTE (Synthetic Minority Over-sampling Technique)
- Transfer learning

## Methodology

### Imports and Paths
- Defined necessary imports and file paths.

### Data Preparation
- Loaded the CSV file and encoded labels.
- Loaded and normalized images.

### K-Fold Cross-Validation
- Set up a stratified K-fold to handle the cross-validation process.

### Training and Evaluation
For each fold:
1. **Data Splitting**: Split the data into training and validation sets.
2. **Data Augmentation**: Applied to the training images using ImageDataGenerator.
3. **Model Selection**: Loaded the selected model and added custom layers.
4. **Model Training**: Compiled and trained the model with early stopping and class weights.
5. **Validation**: Recorded validation accuracy.

### Confusion Matrix
- Made predictions on the validation set.
- Computed and displayed the confusion matrix.

### Advanced Techniques

#### Data Augmentation
- Used ImageDataGenerator to apply random transformations to images, increasing diversity and preventing overfitting.

#### SMOTE Oversampling
- Applied SMOTE to oversample the minority classes, making the dataset more balanced.

#### Transfer Learning with Inception V3
- Leveraged a pre-trained Inception V3 model for feature extraction and added custom layers for classification.

#### Focal Loss
- Used focal loss to focus on hard-to-classify examples and improve performance in minority classes.

#### Early Stopping
- Early stopping is used to prevent overfitting by monitoring validation loss.

### Evaluation Metrics
- Evaluated the model using a classification report and ROC-AUC, providing more insights into performance across classes.

## Data Preprocessing
- Loaded the CSV and encoded labels.
- Normalized images.

## Model Selection

### Inception V3: Overview and Applications

#### Overview
Inception V3, developed by Google researchers, is a deep convolutional neural network architecture introduced in the paper "Rethinking the Inception Architecture for Computer Vision" in 2015. It is designed to achieve high performance on image recognition tasks while maintaining computational efficiency.

#### Key Features
1. **Inception Modules**: Apply multiple convolutional filters of different sizes (1x1, 3x3, 5x5) in parallel to capture features at various scales.
2. **Factorized Convolutions**: Larger convolutions are replaced with multiple smaller convolutions to reduce computational cost.
3. **Auxiliary Classifiers**: Connected to intermediate layers to improve gradient flow and provide regularization.
4. **Batch Normalization**: Applied extensively to accelerate training and improve performance.

#### Architecture
- Composed of multiple inception modules stacked together, followed by fully connected layers and a softmax output layer for classification.
- Typically contains over 42 layers, including convolutions, pooling layers, and inception modules.

#### Applications
1. **Image Classification**: Achieves state-of-the-art performance on benchmarks such as ImageNet.
2. **Feature Extraction**: Used as a feature extractor for other computer vision tasks like object detection and image segmentation.
3. **Transfer Learning**: Pre-trained weights on large datasets like ImageNet make it a popular choice for transfer learning.
4. **Medical Imaging**: Applied in tasks such as detecting diseases in radiology images, analyzing histopathology slides, and segmenting medical images.

#### Advantages
1. **High Accuracy**: Achieves high accuracy on various image classification benchmarks.
2. **Computational Efficiency**: Balances accuracy and computational efficiency using techniques like factorized convolutions.
3. **Scalability**: Adaptable to different computational constraints and performance requirements.

#### Limitations
1. **Complexity**: More complex compared to simpler models like VGG16.
2. **Training Time**: Training from scratch can be time-consuming due to its depth and complexity.

#### Comparison with Other Models
1. **VGG16**: Simpler but less computationally efficient. Inception V3 achieves higher accuracy due to its advanced architecture.
2. **ResNet50**: Uses residual connections to address the vanishing gradient problem. Both models achieve comparable performance but differ in architecture and training dynamics.
3. **EfficientNetB0**: Focuses on scaling and efficiency, often achieving better performance with fewer parameters compared to Inception V3.

## Summary
Inception V3 is a powerful and versatile architecture for image classification and other computer vision tasks, known for its high accuracy and computational efficiency. Its use of inception modules and factorized convolutions makes it a robust choice for a wide range of applications, from academic research to industrial applications.
