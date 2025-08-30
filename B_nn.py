import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib 
from torch.utils.data import TensorDataset, DataLoader
import json 

# --- 配置设备 (CPU 或 GPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. 数据加载与预处理函数 ---
def load_and_preprocess_data(file_path, target_column='target', exclude_columns=['formula']):
    """
    加载CSV数据，删除指定列，并分离特征和目标变量。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    print(f"正在从 {file_path} 加载数据...")
    df = pd.read_csv(file_path)
    print(f"数据加载完成。原始数据形状: {df.shape}")

    if target_column not in df.columns:
        raise ValueError(f"CSV文件中未找到目标列: '{target_column}'")

    for col in exclude_columns:
        if col not in df.columns:
            print(f"警告: 排除列 '{col}' 未在CSV文件中找到，将跳过此列的删除。")

    y_data = df[target_column].values.reshape(-1, 1)
    print(f"目标列 '{target_column}' 已识别。")

    cols_to_drop = [col for col in exclude_columns if col in df.columns]
    if target_column not in cols_to_drop:
        cols_to_drop.append(target_column)

    X_df = df.drop(columns=cols_to_drop, errors='ignore')
    feature_names = X_df.columns.tolist()
    X_data = X_df.values

    print(f"列 '{exclude_columns}' 和 '{target_column}' 已从特征中移除。")
    print(f"处理后的特征数量: {X_data.shape[1]}")

    if X_data.shape[1] != 132:
        print(f"警告: 期望有132个特征，但实际有 {X_data.shape[1]} 个特征。请检查数据列。")

    return X_data, y_data, feature_names

# --- 2. 特征归一化函数 ---
def normalize_features(X_data, scaler=None):
    """
    对输入特征进行标准化归一化处理。
    如果提供了scaler，则使用该scaler进行transform。
    否则，fit_transform并返回新的scaler。
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data)
        print("Features normalized using a new StandardScaler (fit_transform).")
        return X_scaled, scaler
    else:
        X_scaled = scaler.transform(X_data)
        print("Features normalized using an existing StandardScaler (transform only).")
        return X_scaled, scaler

# --- 3. 神经网络模型定义 ---
class RegressionNN(nn.Module):
    def __init__(self, input_size, dropout_rate=0.5): 
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 300)
        self.fc5 = nn.Linear(300, 100)
        self.output_layer = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.dropout5 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.relu(self.fc4(x))
        x = self.dropout4(x)
        x = self.relu(self.fc5(x))
        x = self.dropout5(x)
        x = self.output_layer(x)
        return x

# --- 4. 神经网络模型创建函数 ---
def create_regression_nn(input_size, dropout_rate=0.5): 
    model = RegressionNN(input_size, dropout_rate=dropout_rate).to(device)
    print(f"Regression NN created with input size {input_size}, hidden layers: "
          f"100 -> 300 -> 300 -> 300 -> 100 -> 1, and Dropout Rate: {dropout_rate}. Model moved to {device}.")
    return model

# --- 辅助绘图函数 (保持不变) ---
def plot_dft_vs_ml(true_values, predictions, title, save_path, annotate_metrics=True, color='blue'):
    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predictions, alpha=0.6, color=color)
    
    min_val = min(true_values.min(), predictions.min())
    max_val = max(true_values.max(), predictions.max())
    plot_min = min(min_val, predictions.min(), true_values.min()) - 0.5
    plot_max = max(max_val, predictions.max(), true_values.max()) + 0.5
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', label='Ideal Prediction') 
    
    plt.title(title)
    plt.xlabel('DFT Value (True Target)')
    plt.ylabel('ML Prediction')
    plt.grid(True)
    plt.axis('equal') 
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)

    if annotate_metrics:
        r2 = r2_score(true_values, predictions)
        rmse_val = np.sqrt(mean_squared_error(true_values, predictions))
        text_str = f'$R^2$: {r2:.4f}\nRMSE: {rmse_val:.4f}'
        plt.annotate(text_str, xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8),
                     verticalalignment='top', horizontalalignment='left')

    plt.legend()
    plt.savefig(save_path)
    print(f"{title} plot saved to {save_path}")
    plt.close() 

# --- 绘制训练/验证损失和RMSE曲线 ---
def plot_training_curves(train_losses, val_losses, train_rmses, val_rmses, epochs_trained, save_path, title_prefix=""):
    """
    绘制训练和验证的损失 (MSE) 和 RMSE 曲线。
    """
    # 检查 epochs_trained 是否大于 0，这表明有数据可绘制
    if epochs_trained == 0: 
        print(f"警告: 没有足够的训练数据来绘制 {title_prefix} 曲线。")
        return

    plt.figure(figsize=(12, 6))
    epochs = range(1, epochs_trained + 1) # 使用 epochs_trained 来定义 X 轴范围

    # 绘制损失曲线
    plt.subplot(1, 2, 1) # 1行2列的第一个图
    plt.plot(epochs, train_losses, label='训练损失 (MSE)', color='blue')
    plt.plot(epochs, val_losses, label='验证损失 (MSE)', color='green')
    plt.title(f'{title_prefix} 训练/验证损失曲线')
    plt.xlabel('周期')
    plt.ylabel('损失 (MSE)')
    plt.grid(True)
    plt.legend()

    # 绘制 RMSE 曲线
    plt.subplot(1, 2, 2) # 1行2列的第二个图
    plt.plot(epochs, train_rmses, label='训练 RMSE', color='red')
    plt.plot(epochs, val_rmses, label='验证 RMSE', color='purple')
    plt.title(f'{title_prefix} 训练/验证 RMSE 曲线')
    plt.xlabel('周期')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"{title_prefix} 训练曲线已保存到 {save_path}")
    plt.close()


# --- 训练和评估模型的核心逻辑 ---
def run_training_cycle(X_tensor, y_tensor, model, criterion, optimizer, epochs, batch_size, 
                       eval_X_tensor=None, eval_y_tensor=None, patience=50, min_delta=0.001): 
    
    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_rmse = float('inf') 
    epochs_no_improve = 0 
    best_model_state = None
    best_epoch = 0

    # 用于收集每个epoch的损失和RMSE
    train_losses = []
    val_losses = []
    train_rmses = []
    val_rmses = []

    for epoch in range(epochs):
        model.train()
        current_train_loss_sum = 0.0
        
        # 避免空DataLoader导致错误
        if len(train_loader) == 0:
            print("警告: 训练DataLoader为空，跳过训练循环。")
            break

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            current_train_loss_sum += loss.item() * batch_X.size(0)
        
        avg_epoch_train_loss = current_train_loss_sum / len(train_dataset)
        train_rmse = np.sqrt(avg_epoch_train_loss)

        current_val_rmse = None
        val_loss = float('nan')
        if eval_X_tensor is not None and eval_y_tensor is not None and len(eval_X_tensor) > 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(eval_X_tensor)
                val_loss = criterion(val_preds, eval_y_tensor).item()
                current_val_rmse = np.sqrt(val_loss)
            
            if current_val_rmse is not None:
                if current_val_rmse < best_val_rmse - min_delta: 
                    best_val_rmse = current_val_rmse
                    best_model_state = model.state_dict() 
                    best_epoch = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1} as validation RMSE did not improve for {patience} epochs.")
                    break 
        
        # 记录当前epoch的指标
        train_losses.append(avg_epoch_train_loss)
        val_losses.append(val_loss)
        train_rmses.append(train_rmse)
        val_rmses.append(current_val_rmse if current_val_rmse is not None else float('nan'))

        if (epoch + 1) % 50 == 0 or epoch == epochs - 1 or (current_val_rmse is not None and epochs_no_improve == 0 and epoch > 0):
            log_suffix = f", Current Val RMSE: {current_val_rmse:.4f}" if current_val_rmse else ""
            print(f"Epoch [{epoch+1}/{epochs}]{log_suffix}")
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model from epoch {best_epoch + 1} with Validation RMSE: {best_val_rmse:.4f}")

    # 返回收集的损失和RMSE列表，以及训练好的模型和最佳验证RMSE
    return model, best_val_rmse, train_losses, val_losses, train_rmses, val_rmses

# --- 主要训练函数 (包含交叉验证) ---
def train_with_kfold(X_data, y_data, input_size, k_folds=10, epochs=200, learning_rate=0.001, 
                     batch_size=128, weight_decay=0.0, patience=50, min_delta=0.001, 
                     dropout_rate=0.5, 
                     output_dir="saved_models", scaler_to_save=None, model_prefix="model"): 

    # 仅创建目录，不执行任何清空操作
    os.makedirs(output_dir, exist_ok=True) 
    print(f"Ensuring output directory exists: {output_dir}")

    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_data, dtype=torch.float32).to(device)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    all_fold_best_val_rmses = [] 
    best_overall_val_rmse = float('inf')
    best_overall_model_state = None 

    # 用于平均绘制曲线
    all_folds_train_losses_for_plot = []
    all_folds_val_losses_for_plot = []
    all_folds_train_rmses_for_plot = []
    all_folds_val_rmses_for_plot = []

    print(f"\nStarting {k_folds}-fold cross-validation on {device}...")

    for fold, (train_index, val_index) in enumerate(kf.split(X_tensor)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")

        X_train, X_val = X_tensor[train_index], X_tensor[val_index]
        y_train, y_val = y_tensor[train_index], y_tensor[val_index]

        model = create_regression_nn(input_size, dropout_rate=dropout_rate) 
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # run_training_cycle 现在返回额外的曲线数据
        trained_fold_model_at_best_epoch, current_fold_best_val_rmse, \
        fold_train_losses, fold_val_losses, fold_train_rmses, fold_val_rmses = run_training_cycle(
            X_train, y_train, model, criterion, optimizer, epochs, batch_size,
            eval_X_tensor=X_val, eval_y_tensor=y_val, patience=patience, min_delta=min_delta
        )
        
        all_fold_best_val_rmses.append(current_fold_best_val_rmse) 

        # 收集当前折叠的完整训练历史，用于后续平均和绘图
        all_folds_train_losses_for_plot.append(fold_train_losses)
        all_folds_val_losses_for_plot.append(fold_val_losses)
        all_folds_train_rmses_for_plot.append(fold_train_rmses)
        all_folds_val_rmses_for_plot.append(fold_val_rmses)


        # 更新整体最佳模型
        if current_fold_best_val_rmse < best_overall_val_rmse:
            best_overall_val_rmse = current_fold_best_val_rmse
            best_overall_model_state = trained_fold_model_at_best_epoch.state_dict() # 保存模型状态字典
            print(f"新最佳模型在折叠 {fold+1} 找到，验证RMSE: {best_overall_val_rmse:.4f}")

        print(f"Fold {fold+1} Validation RMSE (best epoch): {current_fold_best_val_rmse:.4f}")

    print("\nCross-validation complete.")

    # 计算平均训练曲线并绘图
    # 确保所有折叠的训练周期数一致，取最短的那个
    min_epochs_trained = min(len(l) for l in all_folds_train_losses_for_plot) if all_folds_train_losses_for_plot else 0
    
    if min_epochs_trained > 0:
        avg_train_losses = np.mean([l[:min_epochs_trained] for l in all_folds_train_losses_for_plot], axis=0)
        avg_val_losses = np.mean([l[:min_epochs_trained] for l in all_folds_val_losses_for_plot], axis=0)
        avg_train_rmses = np.mean([l[:min_epochs_trained] for l in all_folds_train_rmses_for_plot], axis=0)
        avg_val_rmses = np.mean([l[:min_epochs_trained] for l in all_folds_val_rmses_for_plot], axis=0)

        plot_training_curves(avg_train_losses, avg_val_losses, avg_train_rmses, avg_val_rmses,
                             min_epochs_trained, os.path.join(output_dir, "average_kfold_training_curves.png"), 
                             title_prefix="平均 K-Fold")
    else:
        print("警告: 未能绘制平均 K-Fold 训练曲线，因为没有可用的训练数据。")

    # 保存 feature_scaler
    if scaler_to_save is not None:
        scaler_save_path = os.path.join(output_dir, 'feature_scaler.pkl') 
        joblib.dump(scaler_to_save, scaler_save_path)
        print(f"Feature scaler saved to {scaler_save_path}")

    kfold_summary = {
        "k_folds": k_folds,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "dropout_rate": dropout_rate, 
        "early_stopping_patience": patience,
        "early_stopping_min_delta": min_delta,
        "average_validation_rmse_across_folds": np.mean(all_fold_best_val_rmses).item(), 
        "all_fold_best_val_rmses": [rmse.item() for rmse in all_fold_best_val_rmses], 
        "overall_best_validation_rmse_during_kfold": best_overall_val_rmse 
    }
    summary_json_path = os.path.join(output_dir, "kfold_training_summary.json") 
    with open(summary_json_path, 'w') as f:
        json.dump(kfold_summary, f, indent=4)
    print(f"K-Fold训练总结已保存到: {summary_json_path}")

    # 返回最佳模型的状态字典和平均RMSE
    return best_overall_model_state, np.mean(all_fold_best_val_rmses)

# --- 主执行块 ---
if __name__ == "__main__":
    # --- 配置参数 ---
    DATA_BASE_DIR = r"/mnt/c/Users/18769/Desktop/Transform Learning/data" 
    OUTPUT_DIR = r"/mnt/c/Users/18769/Desktop/Transform Learning/NN4B_Direct_Learning_Comparison" 

    TARGET_COL = 'target'
    EXCLUDE_COLS = ['formula']
    INPUT_FEATURES_COUNT = 132 

    try:
        print("\n--- 在 B.csv 数据集上进行 K-Fold 训练 (直接学习) ---")
        
        B_DATA_PATH = os.path.join(DATA_BASE_DIR, "B.csv")
        
        X_b_raw, y_b_raw, _ = load_and_preprocess_data(B_DATA_PATH, TARGET_COL, EXCLUDE_COLS)

        if X_b_raw.shape[1] != INPUT_FEATURES_COUNT:
            raise ValueError(f"B.csv 数据特征数量 ({X_b_raw.shape[1]}) 与预期特征数量 ({INPUT_FEATURES_COUNT}) 不匹配。")
        
        # --- 第一次划分数据集 (80% 训练, 20% 测试) ---
        print("\n--- Splitting B.csv into 80% Training and 20% Final Test Set ---")
        X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(
            X_b_raw, y_b_raw, test_size=0.2, random_state=42, shuffle=True
        )
        print(f"Original B.csv shape: {X_b_raw.shape}")
        print(f"Training set (80%) shape: {X_train_80.shape}")
        print(f"Final Test set (20%) shape: {X_test_20.shape}")

        # 对训练集进行归一化，并保存 scaler
        X_train_80_norm, b_feature_scaler = normalize_features(X_train_80) 
        
        # 对最终测试集进行归一化，使用训练集训练的 scaler
        X_test_20_norm, _ = normalize_features(X_test_20, scaler=b_feature_scaler)

        # K-Fold 训练参数 for B.csv 
        B_KFOLDS = 10 
        B_EPOCHS = 300 
        B_LEARNING_RATE = 0.0001 
        B_BATCH_SIZE = 51 
        B_WEIGHT_DECAY = 0.0 
        B_FINE_TUNE_DROPOUT_RATE = 0.5 

        B_EARLY_STOP_PATIENCE = 50 
        B_EARLY_STOP_MIN_DELTA = 0.001 

        print(f"\n--- Starting K-Fold training on 80% Training Data with parameters: ---")
        print(f"K-Folds: {B_KFOLDS}, Epochs: {B_EPOCHS}, Learning Rate: {B_LEARNING_RATE}, "
              f"Batch Size: {B_BATCH_SIZE}, Weight Decay: {B_WEIGHT_DECAY}")
        print(f"Dropout Rate: {B_FINE_TUNE_DROPOUT_RATE}")
        print(f"Early Stopping Patience: {B_EARLY_STOP_PATIENCE}, Min Delta: {B_EARLY_STOP_MIN_DELTA}")

        # 运行针对 80% 训练数据的 K-Fold 训练
        best_kfold_model_state, avg_val_rmse_across_folds = train_with_kfold(
            X_data=X_train_80_norm, 
            y_data=y_train_80,      
            input_size=INPUT_FEATURES_COUNT,
            k_folds=B_KFOLDS,
            epochs=B_EPOCHS,
            learning_rate=B_LEARNING_RATE,
            batch_size=B_BATCH_SIZE,
            weight_decay=B_WEIGHT_DECAY, 
            patience=B_EARLY_STOP_PATIENCE, 
            min_delta=B_EARLY_STOP_MIN_DELTA, 
            dropout_rate=B_FINE_TUNE_DROPOUT_RATE, 
            output_dir=OUTPUT_DIR, 
            scaler_to_save=b_feature_scaler, 
            model_prefix="B_model_direct" 
        )

        print(f"\nOverall Average Validation RMSE across all {B_KFOLDS} folds (on 80% Training Data): {avg_val_rmse_across_folds:.4f}")
        
        # --- 在 20% 的最终测试集上进行预测和绘图 ---
        print("\n--- Making predictions on the dedicated 20% Final Test Set ---")
        if best_kfold_model_state is not None: 
            # 重新创建模型实例并加载最佳状态
            final_model_for_test_eval = create_regression_nn(INPUT_FEATURES_COUNT, dropout_rate=B_FINE_TUNE_DROPOUT_RATE)
            final_model_for_test_eval.load_state_dict(best_kfold_model_state)
            
            final_model_for_test_eval.eval() 
            with torch.no_grad():
                X_test_20_tensor = torch.tensor(X_test_20_norm, dtype=torch.float32).to(device)
                test_predictions_tensor = final_model_for_test_eval(X_test_20_tensor)
                test_predictions_np = test_predictions_tensor.cpu().numpy().flatten()
                y_test_20_np = y_test_20.flatten()
            
            print(f"Shape of true test values (20% set): {y_test_20_np.shape}")
            print(f"Shape of predicted test values (20% set): {test_predictions_np.shape}")

            plot_dft_vs_ml(y_test_20_np, test_predictions_np, 
                           'DFT Value vs. ML Prediction (20% Dedicated Test Set)', 
                           os.path.join(OUTPUT_DIR, "dft_vs_ml_20_percent_test_predictions.png"), 
                           color='darkorange') 

            # 打印 R2 和 RMSE
            final_test_r2 = r2_score(y_test_20_np, test_predictions_np)
            final_test_rmse = np.sqrt(mean_squared_error(y_test_20_np, test_predictions_np))
            print(f"Final 20% Test Set R^2: {final_test_r2:.4f}")
            print(f"Final 20% Test Set RMSE: {final_test_rmse:.4f}")

            # 保存最终在测试集上表现最佳的模型 (它的状态字典)
            final_best_model_path = os.path.join(OUTPUT_DIR, "best_direct_learning_model_for_B_csv_test_set.pth")
            torch.save(best_kfold_model_state, final_best_model_path)
            print(f"最佳直接学习模型已保存到: {final_best_model_path}")

            # 添加一个总体的JSON summary来包含最终测试集的结果
            final_run_summary_metrics = {
                "direct_learning_hyperparameters": {
                    "learning_rate": B_LEARNING_RATE,
                    "epochs": B_EPOCHS,
                    "batch_size": B_BATCH_SIZE,
                    "dropout_rate": B_FINE_TUNE_DROPOUT_RATE,
                    "weight_decay": B_WEIGHT_DECAY,
                    "early_stopping_patience": B_EARLY_STOP_PATIENCE,
                    "early_stopping_min_delta": B_EARLY_STOP_MIN_DELTA
                },
                "kfold_cross_validation_results": {
                    "num_folds": B_KFOLDS,
                    "average_validation_rmse_across_folds": avg_val_rmse_across_folds,
                    "overall_best_validation_rmse_during_kfold": best_overall_val_rmse 
                },
                "final_test_set_evaluation": {
                    "test_set_fraction": 0.2,
                    "final_test_rmse": final_test_rmse,
                    "final_test_r2": final_test_r2
                }
            }
            summary_json_final_path = os.path.join(OUTPUT_DIR, "direct_learning_final_summary.json")
            with open(summary_json_final_path, 'w') as f:
                json.dump(final_run_summary_metrics, f, indent=4)
            print(f"最终运行总结已保存到: {summary_json_final_path}")

        else:
            print("Error: No model state was returned from K-Fold cross-validation to evaluate on the final test set.")

        print("\nB.csv 训练和评估完成！最终的 20% 测试集预测图、汇总结果和特征归一化器已保存到指定目录。")

        # --- B.csv 数据的分布分析 ---
        print("\n--- Analyzing B.csv Data Distribution for Original Dataset (Informational) ---")
        target_values = y_b_raw.flatten()
        
        plt.figure(figsize=(8, 6))
        bins_all = min(len(np.unique(target_values)) // 2, 20, len(target_values) // 2)
        plt.hist(target_values, bins=max(1, bins_all), density=True, alpha=0.7, color='purple', edgecolor='black')
        plt.title('Original B.csv Target Distribution')
        plt.xlabel('Target Value')
        plt.ylabel('Density')
        plt.grid(True, linestyle='--', alpha=0.6)
        distribution_plot_path = os.path.join(OUTPUT_DIR, "B_csv_original_target_distribution.png") 
        plt.savefig(distribution_plot_path)
        print(f"Original B.csv target distribution plot saved to {distribution_plot_path}")
        plt.close()

        print(f"\nOriginal B.csv Target Statistics:")
        print(f"   Min: {np.min(target_values):.4f}, Max: {np.max(target_values):.4f}")
        print(f"   Mean: {np.mean(target_values):.4f}, Std: {np.std(target_values):.4f}")
        print(f"   Median: {np.median(target_values):.4f}")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()