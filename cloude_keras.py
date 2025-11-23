import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# === Face Detection ===
def detect_face(input_img, cascade_path):
    """Detects the first face in an image, returning the color ROI."""
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar Cascade file not found at: {cascade_path}")

    img_copy = input_img.copy()
    face_cascade = cv2.CascadeClassifier(cascade_path)
    # The model expects color images, so no need to convert to grayscale for the ROI
    faces = face_cascade.detectMultiScale(cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY), scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    (x, y, w, h) = faces[0]
    return img_copy[y:y + h, x:x + w], faces[0]

# === Prepare Training Data for Keras ===
def prepare_training_data(folder_path, cascade_path, target_size=(224, 224)):
    """Loads images, detects faces, and creates labeled training data for Keras."""
    detected_faces = []
    face_labels = []
    
    print("[INFO] Preparing data...")
    person_names = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])

    for person_name in person_names:
        person_folder = os.path.join(folder_path, person_name)
        images_processed_for_folder = 0

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARNING] Could not read image: {image_path}")
                continue

            face, _ = detect_face(image, cascade_path)
            if face is not None:
                # Resize face for MobileNetV2 input size and convert to RGB
                resized_face = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
                resized_face_rgb = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
                detected_faces.append(resized_face_rgb)
                face_labels.append(person_name)
                images_processed_for_folder += 1

        if images_processed_for_folder > 0:
            print(f"[INFO] Processed {images_processed_for_folder} images for '{person_name}'")
        else:
            print(f"[WARNING] No faces detected for '{person_name}'. Skipping.")
            
    return np.array(detected_faces), np.array(face_labels)

# === Build the MobileNetV2 Model ===
def build_model(num_classes, input_shape=(224, 224, 3)):
    """Builds a MobileNetV2-based model for face recognition."""
    # Load the MobileNetV2 model, pre-trained on ImageNet, without the top classification layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the layers of the base model
    base_model.trainable = False

    # Add custom layers on top of the base model
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model

# === Evaluation Function ===
def evaluate_model(model, X_val, y_val_cat, label_encoder):
    """Evaluates the trained Keras model."""
    print("\n" + "="*50)
    print("EVALUATING KERAS MOBILENETV2 MODEL")
    print("="*50)

    # Make predictions
    predictions = model.predict(X_val)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_val_cat, axis=1)

    # Classification Report
    print("\n[CLASSIFICATION REPORT]")
    class_names = label_encoder.classes_
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    accuracy = np.trace(cm) / np.sum(cm)
    plt.title(f'Confusion Matrix | Accuracy: {accuracy:.2f}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# === Prediction Function ===
def predict_single_image(model, image_path, label_encoder, cascade_path, target_size=(224, 224)):
    """Predicts the label for a single test image using the Keras model."""
    test_image = cv2.imread(image_path)
    if test_image is None:
        print(f"[ERROR] Could not load image from {image_path}")
        return

    image_copy = test_image.copy()
    face, rect = detect_face(image_copy, cascade_path)

    if face is None:
        print("[WARNING] No face detected in the test image.")
        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.title("No Face Detected")
        plt.axis("off")
        plt.show()
        return

    # Preprocess the face for the model
    resized_face = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
    resized_face_rgb = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
    face_array = img_to_array(resized_face_rgb)
    face_array = np.expand_dims(face_array, axis=0)
    face_array = face_array / 255.0  # Rescale to [0,1]

    # Predict
    predictions = model.predict(face_array)[0]
    predicted_idx = np.argmax(predictions)
    confidence = predictions[predicted_idx]
    label_text = label_encoder.inverse_transform([predicted_idx])[0]

    print(f"[RESULT] Predicted: '{label_text}' with confidence: {confidence:.2f}")

    # Draw rectangle and text on the image
    (x, y, w, h) = rect
    cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    display_text = f"{label_text} ({confidence*100:.2f}%)"
    cv2.putText(image_copy, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the final image
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.title(f'Prediction: {label_text}')
    plt.axis("off")
    plt.show()

# === Main Script Execution ===
if __name__ == "__main__":
    # --- Configuration ---
    training_data_folder_path = "C:/Users/aiman bawazir/Desktop/onlne_dataset" 
    test_image_path = "C:/Users/aiman bawazir/Desktop/online_dataset/Roger Federer/Roger Federer_6.jpg"
    cascade_path = "C:/Users/aiman bawazir/Desktop/my_fyp/fyp/haarcascade_frontalface_default.xml"
    model_save_path = "C:/Users/aiman bawazir/Desktop/mobilenetv2_face_model.h5"
    label_map_path = "C:/Users/aiman bawazir/Desktop/mobilenet_label_map.json"
    
    # --- Data Preparation ---
    faces, labels = prepare_training_data(training_data_folder_path, cascade_path)

    if len(faces) < 2:
        print("\n[ERROR] Not enough faces detected to train the model. Exiting.")
    else:
        print(f"\n[INFO] Total faces: {len(faces)} | Total subjects: {len(np.unique(labels))}")

        # --- Label Encoding ---
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        num_classes = len(le.classes_)
        labels_cat = to_categorical(labels_encoded, num_classes=num_classes)

        # Save the label mapping
        label_map = {i: label for i, label in enumerate(le.classes_)}
        with open(label_map_path, 'w') as f:
            json.dump(label_map, f, indent=4)
        print(f"[INFO] Label map saved to {label_map_path}")

        # --- Data Preprocessing & Splitting ---
        # Rescale pixel values to the range [0, 1]
        X = faces.astype("float32") / 255.0
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, labels_cat, test_size=0.25, random_state=42, stratify=labels_cat)
        print(f"[INFO] Training samples: {len(X_train)} | Validation samples: {len(X_val)}")
        
        # --- Model Building and Compiling ---
        model = build_model(num_classes)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        
        # --- Model Training ---
        print("\n[INFO] Training Keras MobileNetV2 model...")
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=10,  # Adjust epochs as needed
                            batch_size=32)
        
        model.save(model_save_path)
        print(f"[INFO] Model saved to {model_save_path}")

        # --- Model Evaluation ---
        evaluate_model(model, X_val, y_val, le)
        
        # --- Single Image Prediction ---
        print("\n[INFO] Testing single image prediction...")
        predict_single_image(model, test_image_path, le, cascade_path)
        
    





    # --- For Measuring Inference Time ---
    inference_times = []
    # Load a single test image
    test_image_for_timing = X_val[0] 

    for _ in range(100):  # Repeat 100 times for a stable average
        start_time = time.time()
        # ... your model.predict() or recognizer.predict() call on the test image...
        end_time = time.time()
        inference_times.append(end_time - start_time)

    average_inference_time = sum(inference_times) / len(inference_times)
    print(f"[INFO] Average inference time: {average_inference_time * 1000:.4f} milliseconds")


    # --- For Measuring Model Size ---
    keras_model_size = os.path.getsize("C:/Users/aiman bawazir/Desktop/mobilenetv2_face_model.h5") / (1024 * 1024) # in MB

    print(f"[INFO] MobileNetV2 Model Size: {keras_model_size:.2f} MB")

    print("\n[INFO] Script finished.")