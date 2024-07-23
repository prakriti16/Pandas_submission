# Report
## Approach 
We tried popular image classification techniques like CNN and achieved an accuracy of around 0.8. Further we noticed the distribution of the data across different labels and realised that it is biased, thus we used k-fold cross validation, data augmentation and transfer learning from pre-trained models like Inception V3 to improve performance. 
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
## Data preprocessing
  - Data Preparation: The CSV is loaded, and labels are encoded. Images are loaded and normalized.
## Model Selection
  - We chose Inception V3 due to the following factors:
  - 


