# Predictive Maintenance Framework with Digital Twin Representation (AI4I 2020 Dataset)

This project builds a predictive maintenance framework combining deep learning, pattern recognition, and Asset Administration Shell (AAS)-based device representation using the publicly available AI4I 2020 dataset.

## Objective

To build a reusable end-to-end framework capable of:
- Capturing sensor-based operational data.
- Structuring it in AAS format for interoperability.
- Applying deep learning for pattern recognition and predictive maintenance.
- Visualizing predictions and system health insights.


## Project Structure

- **data/**: Raw dataset from Kaggle.
- **notebooks/**: Final notebook with complete code, models, graphs, and output.
- **aas/**: Generated AAS file representing digital twin format.
- **results/**: Visual outputs, graphs, and reports.
- **docs/**: Project planning and documentation.
- **src/**: Python scripts used for modular execution (e.g., preprocessing, model training).


## Dataset

- Source: [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
- Contains operational metrics of air pressure systems with machine status classification (No Failure, Heat Dissipation Failure, Overstrain, Power Failure, Random Failures).


## Key Technologies Used

- **Python**, **Pandas**, **NumPy**, **Scikit-learn**
- **TensorFlow / Keras** for deep learning models
- **Matplotlib / Seaborn** for visualization
- **AAS JSON format** for digital twin data representation


## Model Features & Outputs

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


## Asset Administration Shell (AAS)

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
    "operational": {
      "u1": 33.8,
      "u2": 47.0,
      "u3": 28.9
    },
    "historical": {
      "type": "L",
      "failures": ["Heat Dissipation Failure"]
    },
    "prediction": {
      "probability_failure": 0.021,
      "status": "Safe"
    }
  }
}
```


## How to Use

1. **Clone the repository**
   ```bash
   git clone https://github.com/zapneel1/predictive-health-maintenence-ai4i-2020.git
   cd predictive-health-maintenence-ai4i-2020
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook notebooks/AI4I_Full_Predictive_Model_With_Safe_Region_with_outputs_final.ipynb
   ```

Or use scripts in `src/` for modular execution.

## Key Outputs

The following outputs were generated during the development and evaluation of the predictive maintenance framework:

1. **Model Performance**
   - Deep learning models were trained over 25 epochs using categorical failure labels.
   - Final test accuracy: approximately **30.4%**, indicating challenges in class imbalance and model generalization.
   - Detailed classification reports were generated for each failure mode:
     - **TWF (Tool Wear Failure)**: Precision/recall near 0 due to under-representation in the dataset.
     - **HDF (Heat Dissipation Failure)**: F1-score around 0.11 for class 1 due to low recall.
     - Majority class (non-failure) was predicted reliably with over 99% accuracy.

2. **Training Progress Plots**
   - Line plots for training and validation loss and accuracy over 25 epochs.
   - The model showed learning progression but also signs of underfitting or poor generalization to minority classes.

3. **Confusion Matrices**
   - Generated confusion matrices for each failure type (TWF, HDF, PWF, OSF, RNF).
   - Highlighted strong class imbalance with almost all predictions biased toward non-failure.

4. **Safe Operating Region Visualization**
   - 3D scatter plots created to illustrate safe zones in the parameter space using features like:
     - Air Temperature
     - Process Temperature
     - Torque, RPM, and Tool Wear
   - This helped define conditions where failure probability was minimal based on prediction results.

5. **Asset Administration Shell (AAS) Output**
   - JSON-based AAS structure generated for each data point, e.g.:
     ```json
     {
       "Asset": {
         "Type": "CNC Machine",
         "Model": "AI4I-2020",
         "Submodels": {
           "OperationalData": {
             "Type": 2,
             "AirTemp[K]": 298.1,
             "ProcessTemp[K]": 308.6,
             "RPM": 1551.0,
             "Torque[Nm]": 42.8,
             "ToolWear[min]": 0.0
           },
           "FailureState": {
             "TWF": 0,
             "HDF": 0,
             "PWF": 0,
             "OSF": 0,
             "RNF": 0
           }
         }
       }
     }
     ```
   - This output supports integration with digital twin platforms and standardization via Industry 4.0 schemas.

6. **Notebook Execution**
   - The full Jupyter Notebook includes:
     - Dataset loading and inspection
     - Model definition and training
     - Evaluation reports and performance metrics
     - Visualizations embedded inline for clear interpretability


## License

This repository is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Dataset by Stephan Matzka on Kaggle
- AAS concept aligned with Industry 4.0 standards
