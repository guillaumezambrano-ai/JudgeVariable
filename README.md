# JudgeVariable
Code for paper : "The Variable Judge", JURIX 2025

# Legal Case Outcome Prediction Pipeline
This project provides a comprehensive, interactive machine learning pipeline designed to predict the outcomes of legal custody cases ("mother", "father", or "shared"). It is built to be flexible and user-friendly, guiding the user through configuration via a command-line interface (CLI) and producing a rich set of analytical outputs.

The pipeline handles the entire workflow from raw data loading to model training, evaluation, and persistence, making it a powerful tool for legal analytics and data science research.

## Key Features

*   **Interactive CLI Setup**: No need to edit code for configuration. The pipeline interactively prompts for file paths, column selections, feature types, and modeling choices.
*   **Reproducibility**: Saves the complete run configuration to a `run_config.json` file. Any experiment can be precisely replicated by loading this file.
*   **Judge-Specific Modeling**: Ability to create separate model "buckets" for judges with sufficient case history, plus a "generic" bucket for all others. This allows the models to capture judge-specific decision patterns.
*   **Flexible Data Balancing**: Addresses class imbalance (e.g., fewer "shared" custody cases) using two distinct methods:
    *   **Class Weighting**: Adjusts the model's loss function to penalize errors on minority classes more heavily.
    *   **Resampling**: Uses `imbalanced-learn` to over-sample minority classes and/or under-sample majority classes.
*   **Robust Model Training & Tuning**:
    *   Trains multiple classifier types: **XGBoost**, **Random Forest**, **Logistic Regression**, and **SVM**.
    *   Uses **Stratified K-Fold Cross-Validation** for reliable performance estimation.
    *   Performs **Randomized Search** for hyperparameter tuning to find the best model configuration.
*   **Advanced Cross-Bucket Evaluation**: After training, it evaluates every model (e.g., the model for Judge A) on every bucket's test set (e.g., the test data for Judge B and the generic group). This helps assess model generalizability.
*   **Comprehensive & Organized Outputs**: Each run creates a timestamped experiment directory containing:
    *   A detailed log file (`pipeline_run.log`).
    *   A multi-sheet Excel report (`pipeline_results.xlsx`) with metrics, hyperparameters, feature importances, and more.
    *   PNG images of all confusion matrices.
    *   Saved model bundles (`.joblib` files) for future use.
*   **Model Persistence**: Each trained model and its corresponding data transformer are saved as a single "bundle". This allows for easy loading and prediction on new, single-case data in other applications.

## Pipeline Workflow

The pipeline follows a logical, multi-step process:

```
[Start]
   |
   V
[1. Configure] -> User provides settings via CLI or loads a `run_config.json`
   |
   V
[2. Load & Merge Data] -> Reads Cases and optional Judges Excel files
   |
   V
[3. Preprocess] -> Maps target variable, imputes missing values
   |
   V
[4. Create Buckets] -> Splits data into Judge-Specific and Generic groups
   |
   V
[5. For each Bucket:]
   |
   +--> [Split Data] -> Create Train/Test sets
   |
   +--> [Fit Transformer] -> Learn scaling/encoding on Train set
   |
   +--> [Cross-Validation Loop on Train Set:]
   |      |
   |      +--> [Balance Fold] -> Apply weighting or resampling
   |      |
   |      +--> [Tune & Train] -> Find best hyperparameters and train model
   |      |
   |      +--> [Evaluate on Validation Fold] -> Calculate metrics
   |
   +--> [Train Final Model] -> On the full (balanced) training set
   |
   +--> [Evaluate on Test Set] -> Assess performance on unseen data
   |
   +--> [Save Model Bundle] -> Persist model and transformer to disk
   |
   V
[6. Cross-Bucket Evaluation] -> Test every saved model on every test set
   |
   V
[7. Export Results] -> Generate detailed XLSX report and all artifacts
   |
   V
[End]
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-folder>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    The necessary libraries are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    If you don't have a `requirements.txt` file, you can create one with the following content or install them directly:
    ```
    pandas
    numpy
    scikit-learn
    xgboost
    imbalanced-learn
    matplotlib
    seaborn
    tabulate
    joblib
    openpyxl # Required by pandas for Excel I/O
    ```

## How to Run

1.  **Prepare your data:**
    *   Ensure you have a **Cases** dataset in an Excel (`.xlsx`) file.
    *   (Optional) Prepare a **Judges** dataset in a separate Excel file if you wish to merge judge-specific features.

2.  **Execute the script from your terminal:**
    ```bash
    python pipeline.py # Assuming the script is named pipeline.py
    ```

3.  **Follow the on-screen prompts:**
    *   **Load Configuration**: The script will first ask if you want to load a configuration from a previous run. This is useful for reproducibility. If you are running for the first time, say no.
    *   **File Paths**: Provide the paths to your cases and judges Excel files.
    *   **Data Merging**: If you provide a judges file, you will be asked to select the columns to link the two datasets (e.g., a judge ID).
    *   **Column Selection**: You will be shown lists of all columns from your merged data. Select the columns to be used as the **Judge ID (for bucketing)**, the **Target Variable** (e.g., `custody_outcome`), and the **Features** for the model. You can select columns by name or by their index in the list.
    *   **Feature Types**: For each feature you selected, confirm whether it is `numeric` (continuous or discrete numbers) or `categorical` (text labels, codes).
    *   **Target Mapping**: Map the raw values in your target column (e.g., "Custody to Mother", "Custody to Father") to the standardized internal classes (`mother`, `father`, `shared`).
    *   **Modeling Choices**: Configure bucketing, class balancing, models to run, and cross-validation settings.

The pipeline will then execute the full workflow and save all results.

## Output Structure

Upon completion, a new directory will be created inside `pipeline_outputs/` with a unique timestamp, for example: `experiment_20231028_153000/`. This directory contains:

*   `run_config.json`: The JSON file containing all the configuration settings for this specific run. You can load this file on a future run to repeat the experiment.
*   `pipeline_run.log`: A detailed log file with timestamps, info, warnings, and errors from the pipeline execution.
*   `pipeline_results.xlsx`: A comprehensive multi-sheet Excel report.
*   `confusion_matrices/`: A folder containing PNG images of confusion matrices for each model and data bucket.
*   `model_bundles/`: A folder containing the serialized model and transformer bundles (`.joblib` files).

### The `pipeline_results.xlsx` Report

This file is the primary output and contains the following sheets:

*   **Distributions**: Shows the class distributions (counts and percentages) for each data split (Train, Test, Folds) before and after balancing.
*   **CV\_Aggregated**: The main cross-validation performance metrics (Accuracy, Precision, Recall, F1-Score) for each model in each bucket, averaged across all folds.
*   **Test\_Metrics**: The final performance metrics of each model on its held-out test set.
*   **Cross\_Test\_Metrics**: The results from the cross-bucket evaluation, showing how well each model performed on every *other* bucket's test set.
*   **Best\_Hyperparameters**: The optimal hyperparameters found by RandomizedSearchCV for each model.
*   **Top\_Features**: The top N most important features for each model, ranked by importance score or coefficient.
*   **Confusion\_Matrices**: The aggregated confusion matrices from both test-set and cross-bucket evaluations, presented in a tabular format.

### Model Bundles (`.joblib`)

The `model_bundles/` directory contains trained models ready for inference. Each `.joblib` file is a Python dictionary containing the fitted model, the fitted data transformer, and metadata.

**Example: Loading and using a model bundle:**
```python
import joblib
import pandas as pd

# Load the bundle
bundle = joblib.load("pipeline_outputs/experiment_.../model_bundles/generic_XGB.joblib")
model = bundle['model']
transformer = bundle['transformer']
meta = bundle['meta']

print(f"Loaded model trained for bucket: {meta['bucket_id']}")

# Create new raw data for prediction (must have the same columns as the training data)
new_case_data = pd.DataFrame([{
    'feature1': 10,
    'feature2': 'CategoryA',
    # ... other features
}])

# 1. Transform the new data using the bundle's transformer
transformed_data = transformer.transform(new_case_data)

# 2. Make a prediction
prediction_numeric = model.predict(transformed_data)
prediction_proba = model.predict_proba(transformed_data)

# Map numeric prediction back to class name
class_names = ["mother", "father", "shared"]
predicted_class = class_names[prediction_numeric[0]]

print(f"Predicted Outcome: {predicted_class}")
print(f"Prediction Probabilities: {prediction_proba}")
```
