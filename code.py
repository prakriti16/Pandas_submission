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
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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

    # Calculate class weights
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
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(train_labels)), activation='softmax')
    ])


    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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

    fold_no += 1

print(f'Average validation accuracy across all folds: {np.mean(all_val_accuracies)}')

test_dir ='/kaggle/input/dataauburn/test_dataset/test_dataset' #change depending on your kaggle directory structure.
# Predict on test set
test_image_paths = [os.path.join(test_dir, img_name) for img_name in os.listdir(test_dir) if img_name.endswith('.jpg')]
results = []
results.append({'File Name': 'File Name', 'Class': 'Class'})

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
output_csv_path='/kaggle/input/outp123/Pandas_submission.csv'
# Save the results to a CSV file
results_df.to_csv(output_csv_path, index=False)
print(f'Predictions saved to {output_csv_path}')

#Just to check the distribution given and predicted 

# Define the path to your CSV file
csv_path = '/kaggle/input/dataauburn/train.csv'  # Adjust the path as needed

# Load the CSV file
df = pd.read_csv(csv_path)

# Count the unique labels
label_counts = df['Class'].value_counts()

# Print the unique labels and their counts
print(label_counts*100/label_counts.sum())

# Define the path to your CSV file
csv_path = '/kaggle/input/outp123/Pandas_submission.csv'  # Adjust the path as needed

# Load the CSV file
df = pd.read_csv(csv_path)

# Count the unique labels
label_counts1 = df['Predicted Label'].value_counts()

# Print the unique labels and their counts
print(label_counts1*100/label_counts1.sum())
