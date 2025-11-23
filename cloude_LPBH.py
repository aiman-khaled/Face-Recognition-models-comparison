import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# === Face Detection ===
def faceDetection(test_img):
    """Detects faces in an image and returns the face regions and grayscale image."""
    # Ensure the Haar Cascade file exists
    haar_cascade_path = r"C:/Users/aiman bawazir/Desktop/my_fyp/fyp/haarcascade_frontalface_default.xml"
    if not os.path.exists(haar_cascade_path):
        raise FileNotFoundError(f"Haar Cascade file not found at: {haar_cascade_path}")
        
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier(haar_cascade_path)
    faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
    return faces, gray_img

# === Prepare Training Data (Auto-Labeling from Folder Names) ===
def labels_for_training_data(directory):
    """Creates face and label lists from a directory of images."""
    faces = []
    faceIDs = []
    label_dict = {}
    current_id = 0

    for folder_name in sorted(os.listdir(directory)):
        folder_path = os.path.join(directory, folder_name)
        if not os.path.isdir(folder_path):
            continue

        images_processed_for_folder = 0
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            test_img = cv2.imread(image_path)
            
            if test_img is None:
                print(f"[WARNING] Failed to load image: {image_path}")
                continue

            faces_rect, gray_img = faceDetection(test_img)
            if len(faces_rect) == 0:
                continue

            # Process only the first detected face
            (x, y, w, h) = faces_rect[0]
            roi_gray = gray_img[y:y+h, x:x+w]
            
            # Resize for consistency
            roi_gray_resized = cv2.resize(roi_gray, (100, 100))
            faces.append(roi_gray_resized)
            faceIDs.append(current_id)
            images_processed_for_folder += 1

        # **CORRECTION**: Only assign a label and increment ID if faces were found
        if images_processed_for_folder > 0:
            label_dict[current_id] = folder_name
            print(f"[INFO] Processed {images_processed_for_folder} images for '{folder_name}' with label {current_id}")
            current_id += 1
        else:
            print(f"[WARNING] No faces detected in any images for '{folder_name}'")
            
    return faces, faceIDs, label_dict

# === Train Classifier ===
def train_classifier(faces, faceIDs):
    """Trains the LBPH face recognizer."""
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceIDs))
    return face_recognizer

# === Evaluation Function ===
def evaluate_model(recognizer, X_val, y_val, label_dict):
    """Evaluates the trained LBPH model on the validation set."""
    if not X_val:
        print("\n[ERROR] Validation set is empty. Cannot evaluate model.")
        return 0, [], []

    print("\n" + "="*50)
    print("EVALUATING LBPH MODEL")
    print("="*50)
    
    y_pred = []
    confidences = []
    
    for face in X_val:
        label, confidence = recognizer.predict(face)
        y_pred.append(label)
        confidences.append(confidence)
    
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\n[RESULTS]")
    print(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification Report
    print(f"\n[CLASSIFICATION REPORT]")
    present_labels = sorted(list(set(y_val) | set(y_pred)))
    present_class_names = [label_dict.get(label, f"Unknown_{label}") for label in present_labels]
    
    print(classification_report(y_val, y_pred, labels=present_labels, 
                                target_names=present_class_names, zero_division=0))
    
    # Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_val, y_pred, labels=present_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=present_class_names,
                yticklabels=present_class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'LBPH Confusion Matrix | Accuracy: {accuracy:.2f}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return accuracy, confidences, y_pred

# === Drawing and Text Functions ===
def draw_rect(test_img, face):
    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def put_text(test_img, text, x, y):
    cv2.putText(test_img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

# === Main Script ===
if __name__ == "__main__":
    # Define paths
    dataset_path = r"C:/Users/aiman bawazir/Desktop/onlne_dataset"
    model_save_path = r'C:/Users/aiman bawazir/Desktop/LPBH_model.yml'
    label_map_path = r'C:/Users/aiman bawazir/Desktop/lbph_label_map.json'
    
    # Load training data
    print("[INFO] Loading and processing dataset...")
    faces, faceIDs, label_dict = labels_for_training_data(dataset_path)
    
    if len(faces) > 1:
        print(f"\n[INFO] Total faces processed: {len(faces)}")
        print(f"[INFO] Total unique subjects: {len(label_dict)}")

        # Save the label mapping
        with open(label_map_path, 'w') as f:
            json.dump(label_dict, f, indent=4)
        print(f"[INFO] Label map saved to {label_map_path}")
        
        # **CORRECTION**: Use standard NumPy array type for images
        X = np.array(faces, dtype=np.uint8)
        y = np.array(faceIDs)
        
        # Split data for training and validation
        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            print("[WARNING] Cannot stratify split (likely too few samples for a class). Using random split.")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
        print(f"[INFO] Training samples: {len(X_train)}")
        print(f"[INFO] Validation samples: {len(X_val)}")
        
        # Train the model
        print("\n[INFO] Training LBPH model...")
        face_recognizer = train_classifier(list(X_train), y_train)
        face_recognizer.save(model_save_path)
        print(f"[INFO] Model saved to {model_save_path}")
        
        # Evaluate the model
        evaluate_model(face_recognizer, list(X_val), y_val, label_dict)
        
        # Test a single image prediction
        print("\n[INFO] Running prediction on a test image...")
        test_img_path = r"C:/Users/aiman bawazir/Desktop/online_dataset/Roger Federer/Roger Federer_6.jpg"
        test_img = cv2.imread(test_img_path)
        
        if test_img is not None:
            faces_detected, gray_img = faceDetection(test_img)
            
            for face in faces_detected:
                (x, y, w, h) = face
                roi_gray = gray_img[y:y+h, x:x+w]
                roi_gray_resized = cv2.resize(roi_gray, (100, 100))
                
                label, confidence = face_recognizer.predict(roi_gray_resized)
                
                predicted_name = label_dict.get(label, "Unknown")
                text_to_display = f"{predicted_name} ({confidence:.2f})"
                
                print(f"[RESULT] Predicted: {predicted_name} | Confidence: {confidence:.2f}")
                
                draw_rect(test_img, face)
                put_text(test_img, text_to_display, x, y-10)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
            plt.title('Single Image Prediction')
            plt.axis('off')
            plt.show()
        else:
            print(f"[ERROR] Could not load test image from: {test_img_path}")
    else:
        print("\n[ERROR] No faces found in the dataset. Training cannot proceed.")

    LPBH_model_size = os.path.getsize("C:/Users/aiman bawazir/Desktop/mobilenetv2_face_model.h5") / (1024 * 1024) # in MB

    print(f"[INFO] MobileNetV2 Model Size: {LPBH_model_size:.2f} MB")

    print("\n[INFO] Script finished.")