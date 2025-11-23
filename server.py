import os
import subprocess
import sys
import threading
from flask import Flask, request, jsonify, send_file

# --- Configuration ---
UPLOAD_FOLDER = 'C:/Users/aiman bawazir/Desktop/oniline_dataset'
MODEL_PATH = 'C:/Users/aiman bawazir/Desktop/mobilenetv2_face_model.tflite'
LABEL_MAP_PATH = 'C:/Users/aiman bawazir/Desktop/mobilenet_label_map.json'
RETRAIN_SCRIPT_PATH = 'C:/Users/aiman bawazir/Desktop/retrain_model.py'
CONVERTER_SCRIPT_PATH = 'C:/Users/aiman bawazir/Desktop/converter.py'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Global state and lock for managing training status ---
training_status = {
    "is_training": False,
    "message": "Not started",
    "new_model_ready": False
}
status_lock = threading.Lock()
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def run_training_scripts():
    """Runs the training and conversion scripts in a background thread."""
    global training_status
    with status_lock:
        training_status.update({"is_training": True, "message": "Step 1/2: Retraining model...", "new_model_ready": False})
    
    try:
        print("\n[BACKGROUND] --- Starting Model Retraining ---")
        subprocess.run([sys.executable, RETRAIN_SCRIPT_PATH], check=True, capture_output=True, text=True)
        print("[BACKGROUND] --- Retraining Complete ---")

        with status_lock:
            training_status['message'] = "Step 2/2: Converting to TFLite..."
        
        print("[BACKGROUND] --- Starting Model Conversion ---")
        subprocess.run([sys.executable, CONVERTER_SCRIPT_PATH], check=True, capture_output=True, text=True)
        print("[BACKGROUND] --- Conversion Complete ---")

        with status_lock:
            training_status.update({"message": "Training complete! New model is ready.", "is_training": False, "new_model_ready": True})
        print("[BACKGROUND] --- All tasks finished successfully! ---\n")

    except subprocess.CalledProcessError as e:
        error_message = f"An error occurred: {e.stderr}"
        with status_lock:
            training_status.update({"message": error_message, "is_training": False, "new_model_ready": False})
        print(f"[BACKGROUND] --- SCRIPT FAILED: {error_message} ---\n")

@app.route('/upload', methods=['POST'])
def upload_face():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({'error': 'Missing image or name in request'}), 400
    image = request.files['image']
    name = request.form['name']
    person_folder = os.path.join(app.config['UPLOAD_FOLDER'], name)
    os.makedirs(person_folder, exist_ok=True)
    filename = f"{name}_{len(os.listdir(person_folder)) + 1}.jpg"
    image.save(os.path.join(person_folder, filename))
    return jsonify({'success': True}), 200

@app.route('/start_training', methods=['POST'])
def start_training():
    with status_lock:
        if training_status['is_training']:
            return jsonify({'message': 'Training is already in progress.'}), 409
        training_thread = threading.Thread(target=run_training_scripts)
        training_thread.start()
    return jsonify({'message': 'Training started in the background.'}), 202

@app.route('/training_status', methods=['GET'])
def get_training_status():
    with status_lock:
        return jsonify(training_status)

@app.route('/get_model', methods=['GET'])
def get_model():
    if not os.path.exists(MODEL_PATH): return jsonify({'error': 'Model not found'}), 404
    return send_file(MODEL_PATH, as_attachment=True)

@app.route('/get_label_map', methods=['GET'])
def get_label_map():
    with status_lock:
        training_status['new_model_ready'] = False
    if not os.path.exists(LABEL_MAP_PATH): return jsonify({'error': 'Label map not found'}), 404
    return send_file(LABEL_MAP_PATH, as_attachment=True)

if __name__ == '__main__':
    print("[INFO] Starting Flask server with background training capability...")
    app.run(host='0.0.0.0', port=5000)