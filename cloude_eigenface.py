import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# === Face Detection ===
def detect_face(input_img, cascade_path):
    """Detects the first face in an image, returning the grayscale ROI."""
    # Ensure the Haar Cascade file exists before use
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar Cascade file not found at: {cascade_path}")

    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    # Return the region of interest (ROI) of the first face found
    (x, y, w, h) = faces[0]
    return gray_img[y:y + h, x:x + w], faces[0]

# === Prepare Training Data ===
def prepare_training_data(folder_path, cascade_path):
    """Loads images, detects faces, and creates labeled training data."""
    detected_faces = []
    face_labels = []
    id_to_name = {}
    current_id = 0

    print("[INFO] Preparing data...")
    for person_name in sorted(os.listdir(folder_path)):
        person_folder = os.path.join(folder_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        images_processed_for_folder = 0
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARNING] Could not read image: {image_path}")
                continue

            face, _ = detect_face(image, cascade_path)
            if face is not None:
                # Resize face for uniform input size
                resized_face = cv2.resize(face, (200, 200), interpolation=cv2.INTER_AREA)
                detected_faces.append(resized_face)
                face_labels.append(current_id)
                images_processed_for_folder += 1
        
        # **CORRECTION**: Only assign a label and increment ID if faces were found
        if images_processed_for_folder > 0:
            print(f"[INFO] Processed {images_processed_for_folder} images for '{person_name}' -> Label {current_id}")
            id_to_name[current_id] = person_name
            current_id += 1
        else:
            print(f"[WARNING] No faces detected for '{person_name}'. Skipping.")
    
    return detected_faces, face_labels, id_to_name

# === Evaluation Function ===
def evaluate_model(recognizer, X_val, y_val, id_to_name):
    """Evaluates the trained model on the validation set."""
    if not X_val:
        print("\n[ERROR] Validation set is empty. Cannot evaluate model.")
        return 0, [], []

    print("\n" + "="*50)
    print("EVALUATING EIGENFACE MODEL")
    print("="*50)
    
    y_pred = []
    confidences = []
    
    for face in X_val:
        label, confidence = recognizer.predict(face)
        y_pred.append(label)
        confidences.append(confidence)
    
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\n[RESULTS]\nValidation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # **CORRECTION**: Use a robust method for getting class names for the report
    present_labels = sorted(list(set(y_val) | set(y_pred)))
    class_names = [id_to_name.get(label, f"Unknown_{label}") for label in present_labels]

    # Classification Report
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_val, y_pred, labels=present_labels, target_names=class_names, zero_division=0))

    # **CORRECTION**: Fixed plotting to show two separate, clean graphs
    # 1. Confidence Distribution Plot
    plt.figure(figsize=(10, 5))
    sns.histplot(confidences, bins=20, kde=True, color='skyblue')
    plt.title('Confidence Score Distribution (Lower is Better)')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # 2. Confusion Matrix Plot
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_val, y_pred, labels=present_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix | Accuracy: {accuracy:.2f}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return accuracy, confidences, y_pred

# === Prediction Function ===
def predict_single_image(recognizer, test_image, id_to_name, cascade_path):
    """Predicts the label for a single test image."""
    image_copy = test_image.copy()
    face, rect = detect_face(image_copy, cascade_path)

    if face is None:
        print("[WARNING] No face detected in the test image.")
        return image_copy, "Unknown"

    resized_face = cv2.resize(face, (200, 200), interpolation=cv2.INTER_AREA)
    label, confidence = recognizer.predict(resized_face)
    label_text = id_to_name.get(label, "Unknown")
    
    print(f"[RESULT] Predicted: '{label_text}' with confidence: {confidence:.2f}")

    # Draw rectangle and text on the image
    (x, y, w, h) = rect
    cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image_copy, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image_copy, label_text

# === Main Script Execution ===
if __name__ == "__main__":
    # --- Configuration ---
    training_data_folder_path = "C:/Users/aiman bawazir/Desktop/onlne_dataset"
    test_image_path = "C:/Users/aiman bawazir/Desktop/online_dataset/Roger Federer/Roger Federer_6.jpg"
    cascade_path = "C:/Users/aiman bawazir/Desktop/my_fyp/fyp/haarcascade_frontalface_default.xml"
    model_save_path = "C:/Users/aiman bawazir/Desktop/eigenfaces_model.yml"
    label_map_path = "C:/Users/aiman bawazir/Desktop/eigen_label_map.json"

    # --- Data Preparation ---
    detected_faces, face_labels, id_to_name = prepare_training_data(training_data_folder_path, cascade_path)

    if len(detected_faces) < 2:
        print("\n[ERROR] Not enough faces detected to train the model. Exiting.")
    else:
        print(f"\n[INFO] Total faces: {len(detected_faces)} | Total subjects: {len(id_to_name)}")

        # Save the label mapping
        with open(label_map_path, 'w') as f:
            json.dump(id_to_name, f, indent=4)
        print(f"[INFO] Label map saved to {label_map_path}")

        # --- Train/Test Split ---
        X = np.array(detected_faces, dtype=np.uint8)
        y = np.array(face_labels)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        print(f"[INFO] Training samples: {len(X_train)} | Validation samples: {len(X_val)}")

        # --- Model Training ---
        print("\n[INFO] Training EigenFace model...")
        eigen_recognizer = cv2.face.EigenFaceRecognizer_create()
        eigen_recognizer.train(X_train, y_train)
        eigen_recognizer.write(model_save_path)
        print(f"[INFO] Model saved to {model_save_path}")

        # --- Model Evaluation ---
        evaluate_model(eigen_recognizer, list(X_val), list(y_val), id_to_name)

        # --- Single Image Prediction ---
        print("\n[INFO] Testing single image prediction...")
        test_image = cv2.imread(test_image_path)
        if test_image is not None:
            predicted_image, predicted_label = predict_single_image(eigen_recognizer, test_image, id_to_name, cascade_path)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB))
            plt.title(f'Prediction: {predicted_label}')
            plt.axis("off")
            plt.show()
        else:
            print(f"[ERROR] Test image not found at: {test_image_path}")
        
    eigen_model_size = os.path.getsize("C:/Users/aiman bawazir/Desktop/eigenfaces_model.yml") / (1024 * 1024) # in MB

    print(f"[INFO] egienface Model Size: {eigen_model_size:.2f} MB")        
    print("\n[INFO] Script finished.")