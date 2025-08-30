import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # Import StandardScaler

# --- Configure device (CPU or GPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Data Loading and Preprocessing Function (unchanged) ---
def load_and_preprocess_data(file_path, target_column='target', exclude_columns=['formula']):
    """
    Loads CSV data, drops specified columns, and separates features and target variable.
    Returns X_data (NumPy array), y_data (NumPy array), feature_names (list)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loading complete. Original data shape: {df.shape}")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV file.")

    y_data = df[target_column].values.reshape(-1, 1)
    
    cols_to_drop = [col for col in exclude_columns if col in df.columns]
    if target_column not in cols_to_drop:
        cols_to_drop.append(target_column)

    X_df = df.drop(columns=cols_to_drop, errors='ignore')
    feature_names = X_df.columns.tolist() # Get feature names
    X_data = X_df.values

    print(f"Processed feature count: {X_data.shape[1]}")
    return X_data, y_data, feature_names

# --- 2. Feature Normalization Function (copied from original training code, as original normalize_features generates scaler) ---
def normalize_features(X_data, scaler_to_use=None):
    """
    Standardizes input features.
    If scaler_to_use is provided, uses it for transformation.
    Otherwise, fit_transforms and returns a new scaler.
    """
    if scaler_to_use:
        X_scaled = scaler_to_use.transform(X_data)
        print("Features normalized using an existing StandardScaler (transform only).")
        return X_scaled
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data)
        print("Features normalized using a new StandardScaler (fit_transform).")
        return X_scaled, scaler


# --- 3. Neural Network Model Definition (copied from original training code, ensuring exact match) ---
class RegressionNN(nn.Module):
    def __init__(self, input_size):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 300)
        self.fc5 = nn.Linear(300, 100)
        self.output_layer = nn.Linear(100, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Debugging: Uncomment these lines if you want to see shapes during forward pass
        x = self.fc1(x)
        # print(f"Shape after fc1: {x.shape}")
        x = self.relu(x)
        # print(f"Shape after relu1: {x.shape}")

        x = self.fc2(x)
        # print(f"Shape after fc2: {x.shape}")
        x = self.relu(x)
        # print(f"Shape after relu2: {x.shape}")

        x = self.fc3(x)
        # print(f"Shape after fc3: {x.shape}")
        x = self.relu(x)
        # print(f"Shape after relu3: {x.shape}")

        x = self.fc4(x)
        # print(f"Shape after fc4: {x.shape}")
        x = self.relu(x)
        # print(f"Shape after relu4: {x.shape}")

        x = self.fc5(x)
        # print(f"Shape after fc5: {x.shape}")
        x = self.relu(x)
        # print(f"Shape after relu5: {x.shape}")

        x = self.output_layer(x)
        # print(f"Shape after output_layer: {x.shape}")
        return x

# --- Main execution block ---
if __name__ == "__main__":
    # --- Configuration Parameters ---
    # !!! Paths are already modified for WSL (Linux) compatibility !!!
    BASE_DIR = "/mnt/c/Users/18769/Desktop/Transform Learning"
    BASE_DATA_DIR = os.path.join(BASE_DIR, "data")
    PRETRAINED_ASSETS_DIR = os.path.join(BASE_DIR, "regression_models")
    SHAP_OUTPUT_DIR = os.path.join(BASE_DIR, "SHAP_Analysis_Formation")
    
    TARGET_COL = 'target'
    EXCLUDE_COLS = ['formula']
    INPUT_FEATURES_COUNT = 132 

    PRETRAINED_MODEL_NAME = 'best_trained_model_weights.pth' 
    PRETRAINED_SCALER_NAME = 'feature_scaler.pkl' 

    PRETRAINED_MODEL_PATH = os.path.join(PRETRAINED_ASSETS_DIR, PRETRAINED_MODEL_NAME)
    PRETRAINED_SCALER_PATH = os.path.join(PRETRAINED_ASSETS_DIR, PRETRAINED_SCALER_NAME)

    FORMATION_DATA_PATH = os.path.join(BASE_DATA_DIR, "formation_data.csv")

    try:
        os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)
        print(f"SHAP analysis results will be saved to: {SHAP_OUTPUT_DIR}")

        print("\n--- SHAP Analysis Phase: for formation_data.csv ---")
        print(f"Pretrained model path: {PRETRAINED_MODEL_PATH}")
        print(f"Scaler path: {PRETRAINED_SCALER_PATH}")
        print(f"Analyzing data source: {FORMATION_DATA_PATH}")

        if not os.path.exists(PRETRAINED_MODEL_PATH):
            raise FileNotFoundError(f"Pretrained model file not found: {PRETRAINED_MODEL_PATH}. Please ensure path is correct and file exists.")
        if not os.path.exists(PRETRAINED_SCALER_PATH):
            raise FileNotFoundError(f"Pretrained Scaler file not found: {PRETRAINED_SCALER_PATH}. Please ensure path is correct and file exists.")

        # 2. Load the pretrained StandardScaler
        loaded_scaler = joblib.load(PRETRAINED_SCALER_PATH)
        print(f"StandardScaler loaded from: {PRETRAINED_SCALER_PATH}")

        # 3. Load and preprocess formation_data.csv data, get feature names
        X_formation_raw, y_formation_raw, feature_names = load_and_preprocess_data(
            FORMATION_DATA_PATH, TARGET_COL, EXCLUDE_COLS
        )

        if X_formation_raw.shape[1] != INPUT_FEATURES_COUNT:
            raise ValueError(f"Feature count in formation_data.csv ({X_formation_raw.shape[1]}) does not match expected feature count ({INPUT_FEATURES_FEATURES}).")

        # 4. Normalize formation_data.csv data using the pretrained StandardScaler
        X_formation_norm = normalize_features(X_formation_raw, scaler_to_use=loaded_scaler)
        
        # Convert NumPy array to PyTorch Tensor and move to device
        X_formation_tensor = torch.tensor(X_formation_norm, dtype=torch.float32).to(device)
        print(f"X_formation_tensor shape for SHAP: {X_formation_tensor.shape}") # Debugging

        # 5. Create model instance and load pretrained weights
        model = RegressionNN(INPUT_FEATURES_COUNT).to(device) 
        
        # --- Debugging: Print Model Architecture and Loaded State Dict Shapes ---
        print("\n--- Debugging: Model Architecture and Loaded State Dict Shapes ---")
        print("Current RegressionNN Architecture:")
        print(model)
        
        # Load state_dict
        # Added weights_only=True for security and to avoid FutureWarning
        state_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=device, weights_only=True) 
        
        print("\nShapes of Tensors in Loaded State Dict:")
        for name, param in state_dict.items():
            print(f"  {name}: {param.shape}")

        # Try to load state_dict
        model.load_state_dict(state_dict)
        print(f"Model weights loaded from: {PRETRAINED_MODEL_PATH}.")
        # --- End Debugging ---

        # 6. Set model to evaluation mode (essential for SHAP analysis)
        model.eval()

        # --- 7. Perform SHAP Analysis ---
        print("\nStarting SHAP Analysis...")
        
        # --- Using SHAP KernelExplainer (Recommended workaround for DeepExplainer issues) ---
        print("\nUsing SHAP KernelExplainer...")
        # KernelExplainer requires NumPy arrays as input
        # Both background data and data to be explained should be NumPy arrays
        X_formation_np = X_formation_tensor.cpu().numpy()

        # KernelExplainer's background data is usually a subset of the training data
        # We randomly select 100 samples as background data.
        # The number of background samples is crucial for KernelExplainer's computation time.
        background_indices_kernel = np.random.choice(X_formation_np.shape[0], 100, replace=False)
        background_data_for_kernel = X_formation_np[background_indices_kernel] 
        print(f"SHAP KernelExplainer background data shape: {background_data_for_kernel.shape}") # Debugging

        # Define a prediction function that takes NumPy array and returns NumPy array
        def model_predict_for_shap(data_np):
            data_tensor = torch.tensor(data_np, dtype=torch.float32).to(device)
            with torch.no_grad(): # No gradients needed for prediction function
                predictions = model(data_tensor)
            return predictions.cpu().numpy().flatten() # Ensure returning a flattened NumPy array

        explainer = shap.KernelExplainer(model_predict_for_shap, background_data_for_kernel)

        print(f"Calculating SHAP values for {X_formation_np.shape[0]} samples. This will be very slow, please be patient...")
        # Calculate SHAP values for all samples
        # For large datasets, you might start with a smaller subset for testing, e.g., shap_values = explainer.shap_values(X_formation_np[:50])
        shap_values = explainer.shap_values(X_formation_np) 
        print("SHAP value calculation complete.")
        
        # Ensure plot_data_X is set correctly for KernelExplainer
        plot_data_X = X_formation_np
        # --- End KernelExplainer Block ---

        # --- Comment out DeepExplainer block if using KernelExplainer ---
        # print("\nStarting SHAP Analysis with DeepExplainer...")
        # background_indices = np.random.choice(X_formation_tensor.shape[0], 100, replace=False)
        # background_data = X_formation_tensor[background_indices] 
        # print(f"SHAP DeepExplainer background data shape: {background_data.shape}") # Debugging

        # explainer = shap.DeepExplainer(model, background_data)
        # print(f"Calculating SHAP values for {X_formation_tensor.shape[0]} samples. This may take a while...")
        # shap_values = explainer.shap_values(X_formation_tensor) 
        # print("SHAP value calculation complete.")
        # plot_data_X = X_formation_tensor.cpu().numpy() # For DeepExplainer, use this

        # 8. Visualize SHAP results and identify top contributing features
        # For regression tasks, shap_values is usually a NumPy array directly, but sometimes returns a list for generality.
        if isinstance(shap_values, list):
            shap_values = shap_values[0] # Take the first element of the list, usually the SHAP values for regression models

        # Calculate mean absolute SHAP values for feature importance
        mean_abs_shap_values = np.abs(shap_values).mean(axis=0)

        # Package feature names and importance into a DataFrame
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Importance': mean_abs_shap_values
        })

        # Sort by importance in descending order
        feature_importance = feature_importance.sort_values(by='SHAP_Importance', ascending=False)

        print("\nFeature Importance (Top 15):")
        print(feature_importance.head(15).to_string()) # Use to_string() to prevent truncation

        # Plot SHAP Bar Plot
        shap.summary_plot(shap_values, plot_data_X, feature_names=feature_names, 
                          plot_type="bar", max_display=15, show=False)
        plt.title('SHAP Bar Plot (Top 15 Features Importance on formation_data.csv)')
        bar_plot_path = os.path.join(SHAP_OUTPUT_DIR, "shap_bar_plot_top15_formation_data.png")
        plt.savefig(bar_plot_path, bbox_inches='tight') # bbox_inches='tight' ensures all content is saved
        print(f"SHAP Bar Plot (Top 15) saved to: {bar_plot_path}")
        plt.close() # Close current figure to free memory

        # Plot SHAP Summary Plot
        shap.summary_plot(shap_values, plot_data_X, feature_names=feature_names, 
                           max_display=15, show=False) # Default is a dot plot
        plt.title('SHAP Summary Plot (Top 15 Features on formation_data.csv)')
        summary_plot_path = os.path.join(SHAP_OUTPUT_DIR, "shap_summary_plot_top15_formation_data.png")
        plt.savefig(summary_plot_path, bbox_inches='tight')
        print(f"SHAP Summary Plot (Top 15) saved to: {summary_plot_path}")
        plt.close()

        print("\nSHAP analysis complete! Please check the generated plots and the printed feature importance list.")

    except FileNotFoundError as e:
        print(f"Error: {e}\nPlease check if the specified pretrained model, Scaler, and data paths are correct.")
    except Exception as e:
        print(f"An error occurred: {e}")