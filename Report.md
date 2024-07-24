# Report
## Approach 
We tried popular image classification techniques like CNN and achieved an accuracy of around 0.7. Further we noticed the distribution of the data across different labels and realised that it is biased, thus we used k-fold cross validation, data augmentation, class weights, Focal Loss, SMOTE and transfer learning to improve performance in terms of AUC-ROC to 0.93 and accuracy upto 0.85. 
  - Imports and Paths: The necessary imports and file paths are defined.
  - Data Preparation: The CSV is loaded, and labels are encoded. Images are loaded and normalized.
  - K-Fold Cross-Validation: A stratified K-fold is set up to handle the cross-validation process.
  - Training and Evaluation:
    - For each fold, the data is split into training and validation sets.
    - Data augmentation is applied to the training images.
    - The selected model is loaded, and custom layers are added.
    - The model is compiled and trained with early stopping and class weights.
    - Validation accuracy is recorded.
  - Confusion Matrix:
    - Predictions on the validation set are made.
    - The confusion matrix is computed and displayed.
  - Data Augmentation: We use ImageDataGenerator to apply random transformations to the images, increasing diversity and preventing overfitting.

  - SMOTE Oversampling: We apply SMOTE to oversample the minority classes, making the dataset more balanced.

  - Transfer Learning with Inception V3: We leverage a pre-trained VGG16 model for feature extraction and add custom layers for classification.

  - Focal Loss: The model uses focal loss to focus on hard-to-classify examples and improve performance on minority classes.

  - Early Stopping: We use early stopping to prevent overfitting by monitoring the validation loss.

  - Evaluation with Metrics Beyond Accuracy: We evaluate the model using a classification report and ROC-AUC, which provide more insights into performance across classes.
## Data preprocessing
  - Data Preparation: The CSV is loaded, and labels are encoded. Images are loaded and normalized.
## Model Selection
  - We chose Inception V3 due to the following factors:
  - Inception V3: Overview and Applications
  - Overview:
    - Inception V3 is a deep convolutional neural network architecture that was introduced as an improvement over the original Inception (GoogLeNet) architecture. Developed by researchers at Google, it is designed to achieve high performance on image recognition tasks while maintaining computational efficiency. Inception V3 is part of the Inception family of models and was introduced in the paper "Rethinking the Inception Architecture for Computer Vision" in 2015.

  - Key Features:

    - Inception Modules: The core component of Inception V3 is the inception module, which applies multiple convolutional filters of different sizes (1x1, 3x3, 5x5) in parallel to the input. This allows the network to capture features at various scales.
    - Factorized Convolutions: Inception V3 employs factorized convolutions, where larger convolutions (e.g., 5x5) are replaced with multiple smaller convolutions (e.g., two 3x3 convolutions) to reduce computational cost.
    - Auxiliary Classifiers: To improve gradient flow during training and provide regularization, Inception V3 includes auxiliary classifiers that are connected to intermediate layers of the network. These classifiers also contribute to the final loss during training.
    - Batch Normalization: Batch normalization is applied extensively throughout the network to accelerate training and improve performance.
  - Architecture:
    - The architecture of Inception V3 is composed of multiple inception modules stacked together, followed by fully connected layers and a softmax output layer for classification. The network typically contains over 42 layers, including convolutions, pooling layers, and inception modules.

  - Applications:

    - Image Classification: Inception V3 is widely used for image classification tasks, where it has achieved state-of-the-art performance on benchmarks such as ImageNet.
    - Feature Extraction: Due to its ability to learn rich and diverse features, Inception V3 is often used as a feature extractor for other computer vision tasks like object detection and image segmentation.
    - Transfer Learning: Inception V3's pre-trained weights on large datasets like ImageNet make it a popular choice for transfer learning. It can be fine-tuned on specific tasks with smaller datasets, leveraging its pre-learned features to improve performance.
    - Medical Imaging: Inception V3 has been applied in medical imaging for tasks such as detecting diseases in radiology images, analyzing histopathology slides, and segmenting medical images.
  - Advantages:

    - High Accuracy: Inception V3 achieves high accuracy on various image classification benchmarks, making it suitable for applications requiring precise image recognition.
    - Computational Efficiency: By using techniques like factorized convolutions and reducing the number of parameters, Inception V3 strikes a balance between accuracy and computational efficiency.
    - Scalability: The architecture can be easily scaled by adding more inception modules, making it adaptable to different computational constraints and performance requirements.
  - Limitations:

    - Complexity: The architecture of Inception V3 is more complex compared to simpler models like VGG16, which can make it harder to implement and tune.
    - Training Time: Training Inception V3 from scratch can be time-consuming due to its depth and complexity, although using pre-trained weights mitigates this issue.
  - Comparison with Other Models:

    - VGG16: VGG16 is simpler and easier to implement but has more parameters, making it less computationally efficient than Inception V3. Inception V3 typically achieves higher accuracy due to its more advanced architecture.
    - ResNet50: ResNet50 uses residual connections to address the vanishing gradient problem and is known for its ease of training and depth. Inception V3 and ResNet50 often achieve comparable performance, but their architectures and training dynamics differ.
    - EfficientNetB0: EfficientNetB0 is designed with a focus on scaling and efficiency, often achieving better performance with fewer parameters compared to Inception V3. It is part of the EfficientNet family, which systematically scales model dimensions.

In summary, Inception V3 is a powerful and versatile architecture for image classification and other computer vision tasks, known for its high accuracy and computational efficiency. Its use of inception modules and factorized convolutions makes it a robust choice for a wide range of applications, from academic research to industrial applications.


