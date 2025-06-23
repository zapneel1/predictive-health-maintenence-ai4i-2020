# Predictive Maintenance Framework with Digital Twin Representation (AI4I 2020 Dataset)

This project builds a predictive maintenance framework combining deep learning, pattern recognition, and Asset Administration Shell (AAS)-based device representation using the publicly available AI4I 2020 dataset.

## 🚀 Objective

To build a reusable end-to-end framework capable of:
- Capturing sensor-based operational data.
- Structuring it in AAS format for interoperability.
- Applying deep learning for pattern recognition and predictive maintenance.
- Visualizing predictions and system health insights.

---

## 📂 Project Structure

- **data/**: Raw dataset from Kaggle.
- **notebooks/**: Final notebook with complete code, models, graphs, and output.
- **aas/**: Generated AAS file representing digital twin format.
- **results/**: Visual outputs, graphs, and reports.
- **docs/**: Project planning and documentation.
- **src/**: Python scripts used for modular execution (e.g., preprocessing, model training).

---

## 📊 Dataset

- Source: [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
- Contains operational metrics of air pressure systems with machine status classification (No Failure, Heat Dissipation Failure, Overstrain, Power Failure, Random Failures).

---

## 🔧 Key Technologies Used

- **Python**, **Pandas**, **NumPy**, **Scikit-learn**
- **TensorFlow / Keras** for deep learning models
- **Matplotlib / Seaborn** for visualization
- **AAS JSON format** for digital twin data representation

---

## 📈 Model Features & Outputs

- Preprocessing: Scaling, missing value imputation, categorical encoding.
- Feature Engineering: Derived features from sensor readings and machine types.
- Models Used:
  - LSTM for time-series sequence prediction.
  - CNN for anomaly classification.
- Performance Metrics:
  - Accuracy, Precision, Recall, F1-Score.
- Safe Operating Region Detection:
  - Visualized regions where machines operate reliably.
- Graphs and Plots:
  - Loss curves, confusion matrix, prediction confidence, and 3D safe zone regions.

---

## 🧠 Asset Administration Shell (AAS)

The sensor data is transformed into a structured AAS representation:
- Submodels include:
  - **Operational Data**
  - **Historical Failures**
  - **Predicted States**

Example output:
```json
{
  "asset": "AirPressureSystem_001",
  "submodels": {
    "operational": {...},
    "prediction": {...},
    "historical": {...}
  }
}
