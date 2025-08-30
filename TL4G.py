import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import json
from torch.utils.data import TensorDataset, DataLoader

# --- 配置类 ---
class Config:
    BASE_DIR = r"/mnt/c/Users/18769/Desktop/Transform Learning/data"
    TARGET_COL = 'target'
    EXCLUDE_COLS = ['formula']
    INPUT_FEATURES_COUNT = 132

    PRETRAIN_MODEL_LOAD_DIR = r"/mnt/c/Users/18769/Desktop/Transform Learning/regression_models"
    PRETRAINED_MODEL_PATH = os.path.join(PRETRAIN_MODEL_LOAD_DIR, 'best_trained_model_weights.pth')
    PRETRAIN_SCALER_PATH = os.path.join(PRETRAIN_MODEL_LOAD_DIR, 'feature_scaler.pkl')

    TRANSFER_DATA_PATH = os.path.join(BASE_DIR, "G.csv")
    
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 设备配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")

# --- 数据加载和预处理 ---
def load_and_preprocess_data(file_path, target_column, exclude_columns):
    df = pd.read_csv(file_path)
    y_data = df[target_column].values.reshape(-1, 1)
    cols_to_drop = [col for col in exclude_columns if col in df.columns]
    if target_column not in cols_to_drop:
        cols_to_drop.append(target_column)
    X_df = df.drop(columns=cols_to_drop, errors='ignore')
    X_data = X_df.values
    return X_data, y_data, X_df.columns.tolist()

def normalize_features(X_data, scaler):
    X_scaled = scaler.transform(X_data)
    return X_scaled

# --- 神经网络模型 ---
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
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

def create_regression_nn(input_size, dropout_rate=0.5):
    model = RegressionNN(input_size, dropout_rate=dropout_rate).to(device)
    return model

# --- 绘图工具 ---
def plot_loss_and_rmse_curves(train_losses, val_losses, train_rmses, val_rmses, epochs_to_plot, save_path, title_prefix=""):
    if not len(train_losses) or not len(val_losses) or not len(train_rmses) or not len(val_rmses):
        return

    plt.figure(figsize=(10, 7))
    x_axis = range(epochs_to_plot)

    plt.plot(x_axis, train_losses[:epochs_to_plot], label='test (MSE)', color='blue', linestyle='-')
    plt.plot(x_axis, val_losses[:epochs_to_plot], label='val (MSE)', color='green', linestyle='-')
    plt.plot(x_axis, train_rmses[:epochs_to_plot], label='test RMSE', color='red', linestyle='-')
    plt.plot(x_axis, val_rmses[:epochs_to_plot], label='val RMSE', color='purple', linestyle='-')

    plt.title(f'{title_prefix} loss and RMSE curves (before {epochs_to_plot} epochs)')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.xlim(0, epochs_to_plot - 1)
    max_val = max(np.max(train_losses), np.max(val_losses), np.max(train_rmses), np.max(val_rmses))
    plt.ylim(0, max(3, max_val * 1.1)) 
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_dft_vs_ml(true_values, predictions, title, save_path, annotate_metrics=True):
    if not len(true_values) or not len(predictions):
        return

    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predictions, alpha=0.6, color='darkorange')

    min_val = min(true_values.min(), predictions.min())
    max_val = max(true_values.max(), predictions.max())
    plot_min = min(min_val, predictions.min(), true_values.min()) - 0.5
    plot_max = max(max_val, predictions.max(), true_values.max()) + 0.5

    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', label='y=x', linewidth=2)

    plt.title(title)
    plt.xlabel('DFT real')
    plt.ylabel('ML prediction')
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

    plt.legend(loc='lower right')
    plt.savefig(save_path)
    plt.close()

# --- 训练和评估 ---
def run_transfer_learning_fold(X_train_fold_tensor, y_train_fold_tensor, X_val_fold_tensor, y_val_fold_tensor, 
                               model, criterion, optimizer, epochs, batch_size, fold_number,
                               patience=50, min_delta=0.001):
    train_dataset = TensorDataset(X_train_fold_tensor, y_train_fold_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    fold_train_losses = []
    fold_val_losses = []
    fold_train_rmses = []
    fold_val_rmses = []

    best_val_rmse = float('inf')
    best_fold_model_state = None
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        current_train_loss_sum = 0.0
        if len(train_loader) == 0:
            break

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            current_train_loss_sum += loss.item() * batch_X.size(0)
        
        avg_epoch_train_loss = current_train_loss_sum / len(train_dataset) if len(train_dataset) > 0 else float('nan')
        train_rmse = np.sqrt(avg_epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = float('nan')
            val_rmse = float('nan')
            if len(X_val_fold_tensor) > 0:
                val_preds = model(X_val_fold_tensor)
                val_loss = criterion(val_preds, y_val_fold_tensor).item()
                val_rmse = np.sqrt(val_loss)

        fold_train_losses.append(avg_epoch_train_loss)
        fold_val_losses.append(val_loss)
        fold_train_rmses.append(train_rmse)
        fold_val_rmses.append(val_rmse)

        if val_rmse < best_val_rmse - min_delta:
            best_val_rmse = val_rmse
            best_fold_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"   折叠 {fold_number} - 早停于周期 {epoch+1}，验证 RMSE 连续 {patience} 周期未改善。")
            break
            
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1 or epochs_no_improve == 0:
            print(f"   折叠 {fold_number} - 周期 [{epoch+1}/{epochs}], 训练损失: {avg_epoch_train_loss:.4f}, 训练RMSE: {train_rmse:.4f}, 验证损失: {val_loss:.4f}, 验证RMSE: {val_rmse:.4f}")

    if best_fold_model_state:
        model.load_state_dict(best_fold_model_state)
    return fold_train_losses, fold_val_losses, fold_train_rmses, fold_val_rmses, best_fold_model_state, best_val_rmse

def evaluate_model_on_test_set(model, X_test_tensor, y_test_raw_np):
    model.eval()
    with torch.no_grad():
        final_predictions_tensor = model(X_test_tensor)
        final_predictions_np = final_predictions_tensor.cpu().numpy().flatten()
    final_rmse = np.sqrt(mean_squared_error(y_test_raw_np, final_predictions_np))
    final_r2 = r2_score(y_test_raw_np, final_predictions_np)
    return final_predictions_np, final_rmse, final_r2


# --- 主执行块 ---
if __name__ == "__main__":
    # --- 超参数设置 ---
    TEST_SET_FRACTION = 0.2
    K_FOLDS = 10
    RANDOM_STATE = 42

    TRANSFER_LEARNING_RATE = 0.0001
    TRANSFER_EPOCHS = 20000
    TRANSFER_BATCH_SIZE = 24
    FINE_TUNE_DROPOUT_RATE = 0.5
    TRANSFER_WEIGHT_DECAY = 0.005

    EARLY_STOP_PATIENCE = 20
    EARLY_STOP_MIN_DELTA = 0.001 

    # 新增超参数：要冻结的隐藏层数
    # 0 表示不冻结任何隐藏层（所有层都微调）
    # 1 表示冻结 fc1
    # 2 表示冻结 fc1, fc2
    # 5 表示冻结 fc1 到 fc5 (即所有隐藏层)
    FREEZE_LAYERS_COUNT = 3 # 默认冻结所有隐藏层，只微调输出层

    OUTPUT_FOLDER_NAME = f"TL4G"

    output_directory_path = os.path.join(Config.CURRENT_SCRIPT_DIR, OUTPUT_FOLDER_NAME)
    os.makedirs(output_directory_path, exist_ok=True)
    print(f"所有输出将保存到目录: {output_directory_path}")

    # 1. 加载和预处理 G.csv 数据
    print("\n--- 开始数据加载和预处理 ---")
    if not os.path.exists(Config.TRANSFER_DATA_PATH):
        raise FileNotFoundError(f"文件未找到: {Config.TRANSFER_DATA_PATH}")
    X_data_raw, y_data_raw, _ = load_and_preprocess_data(
        Config.TRANSFER_DATA_PATH, Config.TARGET_COL, Config.EXCLUDE_COLS
    )
    if X_data_raw.shape[1] != Config.INPUT_FEATURES_COUNT:
        raise ValueError(f"G.csv数据特征数量 ({X_data_raw.shape[1]}) 与预期特征数量 ({Config.INPUT_FEATURES_COUNT}) 不匹配。")

    # 2. 加载预训练的 StandardScaler
    if not os.path.exists(Config.PRETRAIN_SCALER_PATH):
        raise FileNotFoundError(f"预训练的 Scaler 文件未找到: {Config.PRETRAIN_SCALER_PATH}")
    loaded_scaler = joblib.load(Config.PRETRAIN_SCALER_PATH)
    print(f"已加载 StandardScaler: {Config.PRETRAIN_SCALER_PATH}")

    # 3. 使用预训练的 Scaler 对 B.csv 数据进行归一化
    X_data_norm = normalize_features(X_data_raw, scaler=loaded_scaler)

    # 4. 将整体 B.csv 数据分割为训练集和最终测试集
    X_main_train, X_test_final, y_main_train, y_test_final = train_test_split(
        X_data_norm, y_data_raw, test_size=TEST_SET_FRACTION, random_state=RANDOM_STATE
    )

    X_main_train_tensor = torch.tensor(X_main_train, dtype=torch.float32).to(device)
    y_main_train_tensor = torch.tensor(y_main_train, dtype=torch.float32).to(device)
    X_test_final_tensor = torch.tensor(X_test_final, dtype=torch.float32).to(device)
    y_test_final_np = y_test_final.flatten()

    print(f"主训练集大小 (用于K-Fold): {X_main_train.shape[0]} 个样本")
    print(f"最终测试集大小: {X_test_final.shape[0]} 个样本")
    print(f"\n--- 开始在主训练集上进行 {K_FOLDS}-折交叉验证 (微调) ---")
    
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    all_fold_train_losses = []
    all_fold_val_losses = []
    all_fold_train_rmses = []
    all_fold_val_rmses = []
    
    best_overall_val_rmse = float('inf')
    best_overall_tuned_model_state = None

    for fold, (train_index, val_index) in enumerate(kf.split(X_main_train_tensor)):
        print(f"\n--- 正在处理折叠 {fold+1}/{K_FOLDS} ---")

        X_train_fold, X_val_fold = X_main_train_tensor[train_index], X_main_train_tensor[val_index]
        y_train_fold, y_val_fold = y_main_train_tensor[train_index], y_main_train_tensor[val_index]

        transfer_model_fold = create_regression_nn(Config.INPUT_FEATURES_COUNT, dropout_rate=FINE_TUNE_DROPOUT_RATE)
        
        if not os.path.exists(Config.PRETRAINED_MODEL_PATH):
            raise FileNotFoundError(f"预训练模型文件未找到: {Config.PRETRAINED_MODEL_PATH}")
        transfer_model_fold.load_state_dict(torch.load(Config.PRETRAINED_MODEL_PATH, map_location=device))
        print(f"已加载预训练模型权重: {Config.PRETRAINED_MODEL_PATH} 用于折叠 {fold+1} 的微调。")
        
        # --- 冻结层逻辑 ---
        # 隐藏层名称列表，按顺序
        hidden_layer_names = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5']
        
        for name, param in transfer_model_fold.named_parameters():
            param.requires_grad = True # 默认所有参数都可训练
            
            # 判断是否在需要冻结的隐藏层范围内
            for i in range(min(FREEZE_LAYERS_COUNT, len(hidden_layer_names))):
                if hidden_layer_names[i] in name:
                    param.requires_grad = False
                    break # 一旦找到对应的层就跳出
            
            # 确保输出层始终是可训练的 (除非 FREEZE_LAYERS_COUNT 明确覆盖了所有层)
            if "output_layer" in name:
                 param.requires_grad = True

        if FREEZE_LAYERS_COUNT == 0:
            print(f"折叠 {fold+1}: 未冻结任何隐藏层。")
        elif FREEZE_LAYERS_COUNT >= len(hidden_layer_names):
            print(f"折叠 {fold+1}: 已冻结所有 {len(hidden_layer_names)} 个隐藏层。")
        else:
            print(f"折叠 {fold+1}: 已冻结前 {FREEZE_LAYERS_COUNT} 个隐藏层 ({', '.join(hidden_layer_names[:FREEZE_LAYERS_COUNT])})。")
        # --- 冻结层逻辑结束 ---

        transfer_criterion = nn.MSELoss()
        transfer_optimizer = optim.Adam(filter(lambda p: p.requires_grad, transfer_model_fold.parameters()), 
                                        lr=TRANSFER_LEARNING_RATE, weight_decay=TRANSFER_WEIGHT_DECAY)

        fold_train_losses, fold_val_losses, fold_train_rmses, fold_val_rmses, current_fold_best_state, current_fold_best_rmse = \
            run_transfer_learning_fold(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                transfer_model_fold, transfer_criterion, transfer_optimizer, 
                TRANSFER_EPOCHS, TRANSFER_BATCH_SIZE, fold+1,
                patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA
            )
        
        all_fold_train_losses.append(fold_train_losses)
        all_fold_val_losses.append(fold_val_losses)
        all_fold_train_rmses.append(fold_train_rmses)
        all_fold_val_rmses.append(fold_val_rmses)

        if current_fold_best_rmse < best_overall_val_rmse:
            best_overall_val_rmse = current_fold_best_rmse
            best_overall_tuned_model_state = current_fold_best_state
            print(f"在折叠 {fold+1} 中找到新的最佳验证RMSE: {best_overall_val_rmse:.4f}")

    print("\n--- 10-折交叉验证 (微调) 完成 ---")
    print(f"交叉验证中获得的最佳验证RMSE: {best_overall_val_rmse:.4f}")

    min_len = min(len(l) for l in all_fold_train_losses)
    avg_train_losses = np.mean([l[:min_len] for l in all_fold_train_losses], axis=0)
    avg_val_losses = np.mean([l[:min_len] for l in all_fold_val_losses], axis=0)
    avg_train_rmses = np.mean([l[:min_len] for l in all_fold_train_rmses], axis=0)
    avg_val_rmses = np.mean([l[:min_len] for l in all_fold_val_rmses], axis=0)

    plot_loss_and_rmse_curves(
        avg_train_losses, avg_val_losses, avg_train_rmses, avg_val_rmses,
        epochs_to_plot=min_len,
        save_path=os.path.join(output_directory_path, "average_kfold_transfer_learning_curves.png"),
        title_prefix="average K-Fold Transfer Learning"
    )
    
    # 5. 在最终的20%测试集上评估最佳微调模型
    if best_overall_tuned_model_state is not None:
        final_model_for_test_eval = create_regression_nn(Config.INPUT_FEATURES_COUNT, dropout_rate=FINE_TUNE_DROPOUT_RATE)
        
        # --- 应用相同的冻结逻辑到最终评估模型 ---
        hidden_layer_names = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5']
        for name, param in final_model_for_test_eval.named_parameters():
            param.requires_grad = True
            for i in range(min(FREEZE_LAYERS_COUNT, len(hidden_layer_names))):
                if hidden_layer_names[i] in name:
                    param.requires_grad = False
                    break
            if "output_layer" in name:
                 param.requires_grad = True
        # --- 冻结逻辑结束 ---

        final_model_for_test_eval.load_state_dict(best_overall_tuned_model_state)
        print("\n--- 正在最终的20%测试集上评估最佳微调模型 ---")

        final_test_predictions_np, final_test_rmse, final_test_r2 = evaluate_model_on_test_set(
            final_model_for_test_eval, X_test_final_tensor, y_test_final_np
        )

        print(f"\n最佳微调模型在最终测试集上的RMSE: {final_test_rmse:.4f}")
        print(f"最佳微调模型在最终测试集上的R^2: {final_test_r2:.4f}")

        plot_dft_vs_ml(
            y_test_final_np, final_test_predictions_np,
            'DFT value vs. ML prediction (final test set)',
            os.path.join(output_directory_path, "final_test_set_predictions_plot.png")
        )

        final_tuned_model_path = os.path.join(output_directory_path, "best_fine_tuned_model_for_B_csv_test_set.pth")
        torch.save(final_model_for_test_eval.state_dict(), final_tuned_model_path)
        print(f"最佳微调模型已保存到: {final_tuned_model_path}")

        final_run_summary = {
            "transfer_learning_hyperparameters": {
                "learning_rate": TRANSFER_LEARNING_RATE,
                "epochs": TRANSFER_EPOCHS,
                "batch_size": TRANSFER_BATCH_SIZE,
                "dropout_rate": FINE_TUNE_DROPOUT_RATE,
                "weight_decay": TRANSFER_WEIGHT_DECAY,
                "early_stopping_patience": EARLY_STOP_PATIENCE,
                "early_stopping_min_delta": EARLY_STOP_MIN_DELTA,
                "layers_frozen_count": FREEZE_LAYERS_COUNT 
            },
            "kfold_cross_validation_results": {
                "num_folds": K_FOLDS,
                "overall_best_validation_rmse_during_kfold": best_overall_val_rmse,
                "final_validation_rmses_per_fold": [rmses[-1] for rmses in all_fold_val_rmses if rmses]
            },
            "final_test_set_evaluation": {
                "test_set_fraction": TEST_SET_FRACTION,
                "final_test_rmse": final_test_rmse,
                "final_test_r2": final_test_r2
            },
            "pretrained_model_loaded_from": Config.PRETRAINED_MODEL_PATH
        }
        summary_json_path = os.path.join(output_directory_path, "transfer_learning_summary.json")
        with open(summary_json_path, 'w') as f:
            json.dump(final_run_summary, f, indent=4)
        print(f"运行总结已保存到: {summary_json_path}")

    else:
        print("错误: 10折交叉验证后未找到最佳微调模型状态。")

    print(f"\n所有操作已完成！请检查 '{OUTPUT_FOLDER_NAME}' 文件夹以获取所有结果。")