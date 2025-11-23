import tensorflow as tf

# --- Configuration ---
KERAS_MODEL_PATH = 'C:/Users/aiman bawazir/Desktop/mobilenetv2_face_model.h5'
TFLITE_MODEL_PATH = 'C:/Users/aiman bawazir/Desktop/mobilenetv2_face_model.tflite'

print(f"[INFO] Loading Keras model from: {KERAS_MODEL_PATH}")
# Load the full Keras model
model = tf.keras.models.load_model(KERAS_MODEL_PATH)

# Initialize the TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply optimizations (optional but recommended) 
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("[INFO] Converting model to TensorFlow Lite format...")
# Perform the conversion
tflite_model = converter.convert()

print(f"[INFO] Saving TFLite model to: {TFLITE_MODEL_PATH}")
# Save the converted model to a .tflite file
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

print("[INFO] Conversion complete.")