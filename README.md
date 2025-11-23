# Face-Recognition-models-comparison

## 👨‍🔬 Deep Learning-Based Face Recognition System (FYP)

This repository contains the source code for a comprehensive Face Recognition Final Year Project (FYP). The system implements a real-time, low-latency face recognition pipeline utilizing a **MobileNetV2** model converted to **TensorFlow Lite (TFLite)** for deployment on an edge/client device.

A key feature is the **dynamic retraining capability**, allowing new user data to be uploaded via a REST API, triggering an automatic model retraining and serving the updated TFLite model to the client.

-----

## ✨ Project Components and Architecture

The project is split into three main areas: the **Server**, the **Client**, and **Model Development** (including comparative traditional CV models).

### 1\. The Server (Host/Training Machine)

The server is a Flask application designed to manage the dataset and the model lifecycle.

| File | Description |
| :--- | :--- |
| `server/server.py` | **Flask REST API** to handle data uploads (`/upload`), manage background model retraining (`/start_training`), and serve the latest TFLite model and label map to clients. |
| `server/retrain_model.py` | The core script for **MobileNetV2 transfer learning** and training on the collected dataset. It uses Keras and TensorFlow. |
| `server/converter.py` | Converts the trained Keras model (`.h5`) into the smaller, optimized **TFLite** format (`.tflite`) for efficient client deployment. |

### 2\. The Client (Edge/Recognition Device)

| File | Description |
| :--- | :--- |
| `client/temp_gui.py` | **Tkinter-based client application** that uses the TFLite interpreter for real-time face detection and recognition from a webcam feed. It includes functionality to download the latest model and label map from the server. |

### 3\. Model Development & Comparison

These scripts were used for initial setup, detailed analysis, and comparing the deep learning approach against traditional computer vision methods.

| File | Algorithm | Description |
| :--- | :--- | :--- |
| `models/cloude_keras.py` | **MobileNetV2** | Standalone script for initial Keras model setup, training, and detailed evaluation (Classification Report, Confusion Matrix). |
| `models/cloude_LBPH.py` | **LBPH** | Implementation of the **Local Binary Pattern Histograms** traditional face recognition method for comparative analysis. |
| `models/cloude_eigenface.py` | **EigenFace** | Implementation of the **EigenFace** (PCA-based) traditional face recognition method for comparative analysis. |

-----

## 🛠️ Key Requirements

### Hardware / Environment

  * **Host Machine:** (Recommended) A computer with a GPU is best for training the MobileNetV2 model efficiently.
  * **Client Machine:** Any device capable of running Python, OpenCV, and the TFLite interpreter (e.g., a laptop or Raspberry Pi).
  * **Webcam:** Required for the real-time recognition loop on the client.
  * **Network:** The client and server must be on the same network, and the client must be able to reach the server's IP address.

### Dependencies

Install the required Python packages using pip:

```bash
pip install tensorflow opencv-python scikit-learn matplotlib seaborn flask pillow requests
```

-----

## 🚀 Setup and Usage

### 1\. Configuration (Crucial Step)

You must update the hardcoded paths and IP address in the following files before running:

  * **`client/temp_gui.py`**: Change `LAPTOP_SERVER_IP = "192.168.1.101"` to the **actual IP address** of the machine running the Flask server.
  * **All Scripts (`server.py`, `retrain_model.py`, `cloude_*.py`)**: Update all Windows-style paths (e.g., `C:/Users/aiman bawazir/...`) to match your local setup or use relative paths for better portability.

### 2\. Run the Server (Training Host)

Start the Flask server:

```bash
python server/server.py
```

The server will run on port `5000` and wait for upload or training requests.

### 3\. Run the Client (Recognition Device)

Start the GUI application:

```bash
python client/temp_gui.py
```

The client will begin real-time face detection and recognition using the TFLite model loaded from its local path.

### 4\. Dynamic Retraining Workflow

1.  **Upload New Data:** Send a `POST` request to the server's `/upload` endpoint with a face image and a subject name. The server saves the image to the appropriate folder.
2.  **Start Training:** Send a `POST` request to `/start_training`. The server spawns a background thread to run `retrain_model.py` and `converter.py`.
3.  **Monitor Status:** Send a `GET` request to `/training_status` to check the progress.
4.  **Update Client:** On the client GUI, click **"Download Latest Model & Labels"** to fetch the newly generated `.tflite` model and updated label map from the server. The client will then use the updated model for recognition.