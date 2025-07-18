# AI-Powered Anomaly Detection for Network Traffic

This project provides a modular and reusable framework for detecting network anomalies, such as DDoS attacks, using a PyTorch-based Autoencoder model. It has been refactored from its original version to be more robust, flexible, and easier to integrate into other systems.

## Core Concepts

The detection process is split into two main components, following standard MLOps practices:

1.  **The "Kitchen" (`netwok_monitoring_ai/`):** This directory contains the core AI logic. It's where data is processed, models are trained, and the "intelligence" is built. The central piece is the `AnomalyDetector` class.
2.  **The "Restaurant" (`DdosApi/`):** This is a web API (built with FastAPI) that "serves" the trained model. It receives live network traffic data, passes it to the trained model for a prediction, and returns the result.

This separation ensures that the complex and resource-intensive training process is kept separate from the lightweight and fast prediction (inference) process.

## Key Components

-   **`anomaly_detector.py`**: The heart of the project. This class encapsulates the entire ML workflow:
    -   `preprocess_data()`: Converts raw network data (from MikroTik, etc.) into numerical features.
    -   `train()`: Trains the Autoencoder and a simple classifier.
    -   `predict()`: Makes predictions on new, live data.
    -   `save_model()` / `load_model()`: Handles the persistence of all model components (Autoencoder, Classifier, Scaler).

-   **`ml_models/autoencoder_pytorch.py`**: A PyTorch implementation of the Autoencoder model, designed for efficiency and compatibility.

-   **`checkpoints/`**: The directory where the trained models (`.pth` and `.pkl` files) are stored.

## Workflow

### 1. Data Collection

The `DdosApi` is designed to automatically record all incoming traffic data into the `DdosApi/traffic_records/` directory. To build a dataset:

1.  Run the `DdosApi` server.
2.  Run your main application (e.g., MikhPau) and let it send data to the API for a period of time under normal operating conditions.
3.  This will create a collection of JSON files, which will serve as your "normal" dataset.
4.  (Optional) If you can capture traffic during an anomaly or attack, save these files separately.

### 2. Model Training (The "Kitchen")

This process should be done on a machine with the necessary dependencies installed (e.g., a development machine or a dedicated training server).

**a. Setup the Training Environment:**

It is highly recommended to use a Python virtual environment.

```shell
# Navigate to the network_monitoring_ai directory
cd netwok_monitoring_ai

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# .\venv\Scripts\Activate.ps1 # On Windows PowerShell

# Install dependencies
pip install -r requirements.txt 
```
**b. Use the Training Notebook:**

The recommended way to train is by using the provided Jupyter Notebook, which guides you through the process step-by-step.

1.  **Start Jupyter Notebook:**
    From your terminal (with the `venv` activated), run:
    ```shell
    python -m jupyter notebook
    ```
    Your web browser will open with the Jupyter interface.

2.  **Open and Run the Notebook:**
    -   Click on the `Train_Model.ipynb` file.
    -   Inside the notebook, execute each cell in order from top to bottom. You can do this by selecting a cell and pressing **Shift + Enter**.

The notebook will handle loading the data, training the models, and saving the final artifacts (`autoencoder.pth`, `classifier.pkl`, `scaler.pkl`) into the `checkpoints/` directory.

### 3. Deployment

Once the models are trained and saved in the `netwok_monitoring_ai/checkpoints/` directory, the `DdosApi` is ready. Simply restart the `DdosApi` server. It will automatically load the new models on startup and begin providing real predictions.
