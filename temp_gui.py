import tkinter as tk
from tkinter import messagebox, Label, Button
import numpy as np
import cv2
import requests
import json
import os
import threading
from PIL import Image, ImageTk
from tensorflow.lite.python.interpreter import Interpreter

# --- Configuration ---
LAPTOP_SERVER_IP = "192.168.1.101"  # IMPORTANT: Replace with your laptop's ACTUAL IP address
SERVER_URL = f"http://{LAPTOP_SERVER_IP}:5000"
MODEL_PATH = "C:/Users/aiman bawazir/Desktop/mobilenetv2_face_model.tflite"
LABEL_MAP_PATH = "C:/Users/aiman bawazir/Desktop/mobilenet_label_map.json"
CASCADE_PATH = "C:/Users/aiman bawazir/Desktop/my_fyp/fyp/haarcascade_frontalface_default.xml"
TARGET_SIZE = (224, 224)

class FaceRecApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Client (TFLite)")

        # --- Load initial resources ---
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        self.load_model_and_labels()

        # --- State Variables ---
        self.stop_event = threading.Event()
        self.frame_count = 0
        self.last_known_faces = [] # Store last detected faces to reduce flickering
        self.last_prediction_text = "N/A"

        # --- GUI Elements ---
        self.video_label = Label(root)
        self.video_label.pack(pady=10)

        self.prediction_label = Label(root, text="Prediction: N/A", font=("Helvetica", 14, "bold"))
        self.prediction_label.pack(pady=5)

        self.status_label = Label(root, text="Status: Initialized", font=("Helvetica", 12))
        self.status_label.pack(pady=5)
        
        self.update_model_button = Button(root, text="Download Latest Model & Labels", command=self.download_updates)
        self.update_model_button.pack(pady=10, fill=tk.X, padx=20)
        
        self.quit_button = Button(root, text="Quit", command=self.on_closing)
        self.quit_button.pack(pady=5, fill=tk.X, padx=20)

        # --- Start Video Capture and Processing Thread ---
        self.vid = cv2.VideoCapture(0)
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.status_label.config(text="Status: Running Recognition")

    def load_model_and_labels(self):
        """Loads the TFLite model and JSON labels from disk."""
        try:
            with open(LABEL_MAP_PATH, 'r') as f:
                self.label_map = {int(k): v for k, v in json.load(f).items()}
            
            self.interpreter = Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("[INFO] TFLite model and labels loaded successfully.")
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Status: Model and Labels Loaded")
        except Exception as e:
            print(f"[ERROR] Could not load model or labels: {e}")
            messagebox.showerror("Load Error", f"Could not load resources: {e}\nPlease ensure files exist or download updates.")
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Status: Error loading resources!")

    def video_loop(self):
        """Main video processing loop running in a separate thread."""
        while not self.stop_event.is_set():
            ret, frame = self.vid.read()
            if not ret:
                continue

            self.frame_count += 1
            display_frame = frame.copy()

            # --- Performance Optimization: Process every 3rd frame ---
            if self.frame_count % 3 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # --- Tuned parameters for better, more stable detection ---
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.15, 
                    minNeighbors=4, 
                    minSize=(40, 40)
                )
                self.last_known_faces = faces # Update known faces
                
                # If a face is found, run inference
                if len(faces) > 0:
                    # For simplicity, we process only the first (largest) detected face
                    (x, y, w, h) = faces[0]
                    face_roi = frame[y:y + h, x:x + w]
                    
                    # Prepare image for TFLite model
                    resized_face_rgb = cv2.cvtColor(cv2.resize(face_roi, TARGET_SIZE), cv2.COLOR_BGR2RGB)
                    face_array = np.expand_dims(resized_face_rgb, axis=0) / 255.0
                    face_array = np.array(face_array, dtype=np.float32)

                    # Run Inference
                    self.interpreter.set_tensor(self.input_details[0]['index'], face_array)
                    self.interpreter.invoke()
                    predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                    pred_idx = np.argmax(predictions)
                    confidence = predictions[pred_idx]

                    if confidence > 0.5: # Confidence threshold
                        label_text = self.label_map.get(pred_idx, "Unknown")
                        self.last_prediction_text = f"{label_text} ({confidence:.2f})"
                    else:
                        self.last_prediction_text = "Unknown"
                else:
                    self.last_prediction_text = "N/A" # No faces detected
                
                # Update the GUI label for prediction text
                self.prediction_label.config(text=f"Prediction: {self.last_prediction_text}")

            # --- Drawing Optimization: Draw on every frame for smoothness ---
            for (x, y, w, h) in self.last_known_faces:
                # Determine color based on last prediction
                color = (0, 255, 0) if "Unknown" not in self.last_prediction_text and "N/A" not in self.last_prediction_text else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            # Convert frame for Tkinter display
            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

    def download_updates(self):
        """Downloads both the model and the label map from the server."""
        self.status_label.config(text="Status: Downloading updates...")
        self.root.update_idletasks() # Force GUI to update text

        try:
            # Download Model
            print("[INFO] Requesting model from server...")
            model_response = requests.get(f"{SERVER_URL}/get_model", timeout=30)
            model_response.raise_for_status() # Raises an error for bad status codes (4xx or 5xx)

            # Download Label Map
            print("[INFO] Requesting label map from server...")
            labels_response = requests.get(f"{SERVER_URL}/get_label_map", timeout=10)
            labels_response.raise_for_status()

            # If both downloads succeed, save the files
            with open(MODEL_PATH, 'wb') as f:
                f.write(model_response.content)
            print(f"[INFO] New model saved to {MODEL_PATH}")

            with open(LABEL_MAP_PATH, 'wb') as f:
                f.write(labels_response.content)
            print(f"[INFO] New label map saved to {LABEL_MAP_PATH}")

            # Reload the model and labels into the application
            self.load_model_and_labels()
            messagebox.showinfo("Success", "New model and labels downloaded and loaded successfully!")

        except requests.exceptions.RequestException as e:
            messagebox.showerror("Download Error", f"Failed to connect or download from server: {e}")
            self.status_label.config(text="Status: Download Failed!")

    def on_closing(self):
        """Cleanly closes the application."""
        print("[INFO] Closing application...")
        self.stop_event.set()
        # Wait a moment for the thread to see the event
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.vid.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecApp(root)
    root.mainloop()