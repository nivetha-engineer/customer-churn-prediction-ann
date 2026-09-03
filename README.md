# Customer Churn Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview

This project predicts whether a telecom customer is likely to **churn** (leave the service) using an **Artificial Neural Network (ANN)**.

The project follows an end-to-end machine learning workflow:

- Data loading and exploration
- Data cleaning and preprocessing
- Categorical feature encoding
- Numerical feature scaling
- Train-test split
- ANN model building
- Model training with Early Stopping
- TensorBoard monitoring
- Training performance visualization
- Precision-Recall analysis
- F1-score based decision-threshold selection
- Classification report and confusion matrix
- Saving and loading the trained model
- Prediction on a new/unseen customer

## 🎯 Objective

The main objective is to identify customers who are likely to churn so that a business can take preventive actions such as targeted offers, better support, or retention campaigns.

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset.

The dataset contains customer information such as:

- Gender
- Senior citizen status
- Partner and dependents
- Tenure
- Phone service
- Internet service
- Online security and backup
- Device protection
- Technical support
- Streaming services
- Contract type
- Paperless billing
- Payment method
- Monthly charges
- Total charges

### Target Variable

`Churn`

- `Yes` → Customer churned
- `No` → Customer did not churn

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- TensorBoard
- Pickle
- Jupyter Notebook

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Feature / Target Separation
   ↓
Label Encoding of Target
   ↓
Train-Test Split
   ↓
One-Hot Encoding + Standard Scaling
   ↓
ANN Model
   ↓
Model Training
   ↓
Early Stopping + TensorBoard
   ↓
Prediction Probabilities
   ↓
Precision-Recall Analysis
   ↓
F1-Score Based Threshold
   ↓
Classification Report
   ↓
Confusion Matrix
   ↓
Save Model + Preprocessor + Label Encoder
   ↓
Prediction on New Customer
```

## 🧹 Data Preprocessing

### 1. TotalCharges Conversion

`TotalCharges` is converted to a numeric datatype. Invalid values are converted to missing values and then filled using the mean.

### 2. Remove Customer ID

`customerID` is removed because it is an identifier and is not useful for prediction.

### 3. Target Encoding

The `Churn` target is converted from text labels to binary values using `LabelEncoder`.

### 4. Feature Encoding

Categorical features are converted using `OneHotEncoder(drop='first')`.

### 5. Feature Scaling

Numerical features are standardized using `StandardScaler`.

A `ColumnTransformer` combines both preprocessing operations.

## 🧠 ANN Architecture

The neural network used in this project contains:

```text
Input Layer
    ↓
Dense Layer: 64 neurons + ReLU
    ↓
Dropout: 20%
    ↓
Dense Layer: 32 neurons + ReLU
    ↓
Dropout: 20%
    ↓
Dense Layer: 16 neurons + ReLU
    ↓
Dropout: 20%
    ↓
Output Layer: 1 neuron + Sigmoid
```

### Model Configuration

- Optimizer: Adam
- Loss function: Binary Crossentropy
- Metric: Accuracy
- Batch size: 32
- Maximum epochs: 100
- Early stopping patience: 5
- Restore best weights: Enabled

## 📈 Model Evaluation

The project evaluates the model using:

- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix
- Precision-Recall curve

Instead of relying only on the default probability threshold of `0.50`, the notebook calculates an F1-score across possible thresholds and selects the threshold that gives the highest F1-score.

This is useful for customer churn because identifying potential churners correctly can be more important than simply maximizing accuracy.

## 📊 TensorBoard

TensorBoard is used to monitor training performance.

The notebook records training information under:

```text
logs/fit/
```

You can launch TensorBoard from Jupyter using:

```python
%load_ext tensorboard
%tensorboard --logdir ./logs
```

## 💾 Saved Model Files

The trained components are saved as:

```text
ann_churn_model.keras
preprocessor.pkl
label_encoder.pkl
```

### `ann_churn_model.keras`

Contains the trained ANN model.

### `preprocessor.pkl`

Contains the fitted `ColumnTransformer`, including:

- One-hot encoding
- Standard scaling

### `label_encoder.pkl`

Contains the fitted target `LabelEncoder` used to convert the prediction back to `Yes` / `No`.

## 🔮 Example Prediction

The notebook creates an unseen customer profile and performs prediction after loading the saved preprocessing objects and ANN model.

The prediction provides:

```text
Churn Probability
Customer Churn Status
```

## 📁 Recommended Repository Structure

```text
customer-churn-prediction-ann/
│
├── Customer_Churn_ANN.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── ann_churn_model.keras
│   ├── preprocessor.pkl
│   └── label_encoder.pkl
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
└── logs/
    └── ...
```

> **Note:** Large datasets, TensorBoard logs, and generated model files do not always need to be committed to GitHub. For a portfolio repository, it is often better to keep the repository lightweight and explain where the dataset/model files come from.

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/nivetha-engineer/customer-churn-prediction-ann.git
cd customer-churn-prediction-ann
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebook

```bash
jupyter notebook
```

Then open:

```text
Customer_Churn_ANN.ipynb
```

Run the cells from top to bottom.

## 📦 Requirements

The main libraries required are:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- jupyter
- tensorboard

See `requirements.txt` for the package list.


## 🚀 Possible Future Improvements

- Add ROC-AUC and PR-AUC metrics
- Compare ANN with Logistic Regression, Random Forest, XGBoost, and SVM
- Use cross-validation
- Tune ANN hyperparameters
- Handle class imbalance with appropriate techniques
- Build a Streamlit web application
- Deploy the model using a cloud platform
- Save the selected threshold along with the model artifacts
- Add a prediction API using Flask or FastAPI

## 👩‍💻 Author

**Nivetha**

GitHub: https://github.com/nivetha-engineer

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.
