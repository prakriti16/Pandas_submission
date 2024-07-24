from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score

# Define paths
data_dir = '/kaggle/input/dataauburn/train_dataset/train_dataset'
csv_path = '/kaggle/input/dataauburn/train.csv'  # Adjust the path as needed

# Load CSV file
df = pd.read_csv(csv_path)

# Encode the labels
label_encoder = LabelEncoder()
df['encoded_class'] = label_encoder.fit_transform(df['Class'])

# Define image size and number of channels
image_height, image_width, num_channels = 128, 128, 3

# Function to load images from dataframe
def load_images_from_dataframe(df, directory):
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(directory, row['File Name'])
        img = image.load_img(img_path, target_size=(image_height, image_width))
        img_array = image.img_to_array(img)
        images.append(img_array)
        labels.append(row['encoded_class'])
    return np.array(images), np.array(labels)

# K-Fold Cross-Validation
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
fold_no = 1
all_val_accuracies = []

for train_index, val_index in kfold.split(df, df['encoded_class']):
    train_df = df.iloc[train_index]
    val_df = df.iloc[val_index]

    # Load train and validation images
    train_images, train_labels = load_images_from_dataframe(train_df, data_dir)
    val_images, val_labels = load_images_from_dataframe(val_df, data_dir)

    # Normalize the pixel values
    train_images = train_images / 255.0
    val_images = val_images / 255.0

    # Use SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(train_images.reshape(train_images.shape[0], -1), train_labels)
    train_images = X_resampled.reshape(-1, image_height, image_width, num_channels)
    train_labels = y_resampled

    # Calculate class weights (optional with SMOTE but might still be useful)
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = dict(enumerate(class_weights))

    # Create an instance of ImageDataGenerator for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Fit the data generator on the training data
    datagen.fit(train_images)

    # Load the selected model with pre-trained weights
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_height, image_width, num_channels))
    
    # Freeze the base model
    base_model.trainable = False

    # Add custom layers on top of the base model
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),  # Increased layer size for better feature extraction
        tf.keras.layers.Dropout(0.5),  # Added dropout for regularization
        tf.keras.layers.Dense(len(np.unique(train_labels)), activation='softmax')
    ])

    # Define Focal Loss function
    def focal_loss(gamma=2.0, alpha=0.25):
        def focal_loss_fixed(y_true, y_pred):
            # Convert to logits
            y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
            y_true = tf.cast(y_true, tf.int32)

            # Convert y_true to one-hot encoding
            y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])

            # Calculate focal loss components
            cross_entropy_loss = -y_true_one_hot * tf.math.log(y_pred)
            loss = alpha * tf.pow((1 - y_pred), gamma) * cross_entropy_loss

            return tf.reduce_mean(loss)

        return focal_loss_fixed

    # Compile the model
    model.compile(optimizer='adam', loss=focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])

    # Define the early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model with data augmentation and class weights
    print(f'Training for fold {fold_no} ...')
    model.fit(datagen.flow(train_images, train_labels, batch_size=32),
              validation_data=(val_images, val_labels),
              epochs=50,
              callbacks=[early_stopping],
              class_weight=class_weights)

    # Evaluate the model
    val_loss, val_acc = model.evaluate(val_images, val_labels)
    all_val_accuracies.append(val_acc)
    print(f'Validation accuracy for fold {fold_no}: {val_acc}')

    # Compute the confusion matrix
    val_predictions = model.predict(val_images)
    val_predictions = np.argmax(val_predictions, axis=1)
    cm = confusion_matrix(val_labels, val_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for fold {fold_no}')
    plt.show()

    # Classification report and ROC-AUC
    print(classification_report(val_labels, val_predictions, target_names=label_encoder.classes_))
    roc_auc = roc_auc_score(tf.keras.utils.to_categorical(val_labels), tf.keras.utils.to_categorical(val_predictions), multi_class='ovr')
    print(f'AUC-ROC for fold {fold_no}: {roc_auc}')

    fold_no += 1

print(f'Average validation accuracy across all folds: {np.mean(all_val_accuracies)}')

# Test set predictions
test_dir = '/kaggle/input/dataauburn/test_dataset/test_dataset'  # Adjust path as needed
test_image_paths = [os.path.join(test_dir, img_name) for img_name in os.listdir(test_dir) if img_name.endswith('.jpg')]
results = []

def predict_image(model, img_path, label_encoder):
    img = image.load_img(img_path, target_size=(image_height, image_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

for img_path in test_image_paths:
    predicted_label = predict_image(model, img_path, label_encoder)
    img_name = os.path.basename(img_path)
    results.append({'File Name': img_name, 'Class': predicted_label})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)
output_csv_path = '/kaggle/working/Pandas_submission.csv'  # Change path if needed
# Save the results to a CSV file
results_df.to_csv(output_csv_path, index=False)
print(f'Predictions saved to {output_csv_path}')

