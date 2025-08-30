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
import joblib # 用于保存和加载Scaler
import json # 用于保存超参数和结果

# --- 设备配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# --- 1. 数据加载和预处理函数 ---
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
        raise ValueError(f"目标列 '{target_column}' 未在CSV文件中找到。")

    for col in exclude_columns:
        if col not in df.columns:
            print(f"警告: 排除列 '{col}' 未在CSV中找到，已跳过。")

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
        print(f"警告: 预期132个特征，但找到 {X_data.shape[1]}。请检查数据列。")

    return X_data, y_data, feature_names

# --- 2. 特征归一化函数 (修改后的版本，支持fit和transform分离) ---
def create_feature_scaler(X_train_data):
    """
    使用训练数据初始化并拟合StandardScaler。
    """
    scaler = StandardScaler()
    scaler.fit(X_train_data) # **只在训练数据上拟合**
    print("StandardScaler已在训练数据上拟合。")
    return scaler

def transform_features(X_data, scaler):
    """
    使用预训练的Scaler转换输入特征。
    """
    X_scaled = scaler.transform(X_data)
    print("特征已使用StandardScaler转换。")
    return X_scaled

# --- 3. 神经网络模型定义 (带Dropout) ---
class RegressionNN(nn.Module):
    def __init__(self, input_size, dropout_rate=0.1):
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

# --- 4. 神经网络模型创建函数 ---
def create_regression_nn(input_size, dropout_rate):
    """
    创建并返回一个回归NN模型，并将其移动到指定设备。
    """
    model = RegressionNN(input_size, dropout_rate=dropout_rate).to(device)
    print(f"回归NN已创建，输入大小 {input_size}，Dropout Rate {dropout_rate}，隐藏层: "
          f"100 -> 300 -> 300 -> 300 -> 100 -> 1。模型已移动到 {device}。")
    return model

# --- 绘图工具 ---
def plot_loss_and_rmse_curves(train_losses, train_rmses, val_losses, val_rmses, epochs_to_plot, save_path, title_prefix=""):
    """Plots training and validation loss and RMSE curves."""
    if not len(train_losses) or not len(val_losses) or not len(train_rmses) or not len(val_rmses):
        print(f"Warning: Cannot plot curves as data is empty.")
        return

    plt.figure(figsize=(10, 7))
    x_axis = range(epochs_to_plot)

    plt.plot(x_axis, train_losses[:epochs_to_plot], label='Training Loss (MSE)', color='blue', linestyle='-')
    plt.plot(x_axis, val_losses[:epochs_to_plot], label='Validation Loss (MSE)', color='green', linestyle='-')
    plt.plot(x_axis, train_rmses[:epochs_to_plot], label='Training RMSE', color='red', linestyle='-')
    plt.plot(x_axis, val_rmses[:epochs_to_plot], label='Validation RMSE', color='purple', linestyle='-')

    plt.title(f'{title_prefix} Loss and RMSE Curves (First {epochs_to_plot} Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.xlim(0, epochs_to_plot - 1)
    # Dynamically adjust Y-axis limit for better visualization, ensure it's at least 3
    max_val = max(np.max(train_losses), np.max(val_losses), np.max(train_rmses), np.max(val_rmses))
    plt.ylim(0, max(3, max_val * 1.1)) 
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(save_path)
        print(f"Loss and RMSE curves saved to {save_path}")
    except Exception as e:
        print(f"Error: Could not save curves plot to {save_path}. Reason: {e}")
    finally:
        plt.close()

def plot_dft_vs_ml(true_values, predictions, title, save_path, annotate_metrics=True):
    """Plots DFT (True Target) vs. ML Predictions."""
    if not len(true_values) or not len(predictions):
        print(f"Warning: Cannot plot DFT vs. ML for '{title}' as true or predicted values are empty.")
        return

    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predictions, alpha=0.6, color='darkorange')

    min_val = min(true_values.min(), predictions.min())
    max_val = max(true_values.max(), predictions.max())
    # Adjust plot limits based on actual data range
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

    plt.legend(loc='lower right')
    try:
        plt.savefig(save_path)
        print(f"'{title}' plot saved to {save_path}")
    except Exception as e:
        print(f"Error: Could not save '{title}' plot to {save_path}. Reason: {e}")
    finally:
        plt.close()

# --- 5. 训练和评估函数 ---
def run_single_fold_training_cycle(X_train_fold_tensor, y_train_fold_tensor, X_val_fold_tensor, y_val_fold_tensor, 
                                   model, criterion, optimizer, epochs, batch_size, base_save_dir, fold_number):
    """
    执行单个折叠的训练循环，包括验证和保存中间图/模型。
    此版本旨在为单个最终运行提供详细输出。
    """
    train_dataset = torch.utils.data.TensorDataset(X_train_fold_tensor, y_train_fold_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    fold_train_losses = []
    fold_val_losses = []
    fold_train_rmses = []
    fold_val_rmses = []

    best_val_rmse = float('inf')
    best_model_state = None
    
    print(f"正在保存折叠特定输出到: {base_save_dir}")

    for epoch in range(epochs):
        # 训练
        model.train()
        current_train_loss_sum = 0.0
        if len(train_loader) == 0:
            print("警告: 训练数据加载器为空。跳过此折叠的剩余epoch。")
            break

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            current_train_loss_sum += loss.item() * batch_X.size(0)
        
        if len(train_dataset) > 0:
            avg_epoch_train_loss = current_train_loss_sum / len(train_dataset)
            train_rmse = np.sqrt(avg_epoch_train_loss)
        else:
            avg_epoch_train_loss = float('nan')
            train_rmse = float('nan')

        # 验证
        model.eval()
        with torch.no_grad():
            if len(X_val_fold_tensor) > 0:
                val_preds = model(X_val_fold_tensor)
                val_loss = criterion(val_preds, y_val_fold_tensor).item()
                val_rmse = np.sqrt(val_loss)
            else:
                val_loss = float('nan')
                val_rmse = float('nan')

        fold_train_losses.append(avg_epoch_train_loss)
        fold_val_losses.append(val_loss)
        fold_train_rmses.append(train_rmse)
        fold_val_rmses.append(val_rmse)

        # 根据验证RMSE保存此折叠的最佳模型状态
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict()
            # 直接保存到主目录，文件名包含折叠号
            torch.save(model.state_dict(), os.path.join(base_save_dir, f'best_model_fold_{fold_number}.pth'))
            print(f"    已保存此折叠的最佳模型状态。验证RMSE: {best_val_rmse:.4f}")

        # 每10个epoch或最后一个epoch打印详细日志
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {avg_epoch_train_loss:.4f}, Train RMSE: {train_rmse:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")
    
    # 此折叠训练完成后，为该折叠的验证集生成DFT vs ML图
    if len(X_val_fold_tensor) > 0:
        model.eval() # 确保模型处于评估模式
        with torch.no_grad():
            val_predictions_np = model(X_val_fold_tensor).cpu().numpy().flatten()
        
        plot_dft_vs_ml(
            y_val_fold_tensor.cpu().numpy().flatten(), val_predictions_np,
            f'DFT vs ML Prediction (Fold {fold_number} Validation)', # 标题包含折叠号
            os.path.join(base_save_dir, f"dft_vs_ml_fold_{fold_number}_validation.png") # 文件名包含折叠号
        )

    return fold_train_losses, fold_val_losses, fold_train_rmses, fold_val_rmses, best_model_state, best_val_rmse

def evaluate_model_on_test_set(model, X_test_tensor, y_test_raw_np):
    """评估模型在最终保留测试集上的性能，并返回预测和指标。"""
    model.eval()
    with torch.no_grad():
        final_predictions_tensor = model(X_test_tensor)
        final_predictions_np = final_predictions_tensor.cpu().numpy().flatten()

    final_rmse = np.sqrt(mean_squared_error(y_test_raw_np, final_predictions_np))
    final_r2 = r2_score(y_test_raw_np, final_predictions_np)
    return final_predictions_np, final_rmse, final_r2


# --- 主执行块 ---
if __name__ == "__main__":
    # --- 可配置超参数 ---
    DATA_FILE_PATH = r"/mnt/c/Users/18769/Desktop/Transform Learning/data/formation_data.csv"
    TARGET_COLUMN = 'target'
    EXCLUDE_COLUMNS = ['formula']
    INPUT_FEATURES_EXPECTED = 132 # 预处理后预期的特征数量

    # 本次运行的固定超参数
    FIXED_LEARNING_RATE = 0.001
    FIXED_BATCH_SIZE = 64
    FIXED_DROPOUT_RATE = 0.2
    FIXED_EPOCHS = 100

    TEST_SET_FRACTION = 0.2 # 用于最终测试集的数据比例（例如，0.2表示20%）
    K_FOLDS = 10 # 训练数据上的交叉验证折叠数
    RANDOM_STATE = 12 # 用于数据分割的随机种子，以保证可复现性

    # 所有结果的输出目录
    BASE_SAVE_DIRECTORY = "regression_models" # 更改目录名以反映单次运行
    os.makedirs(BASE_SAVE_DIRECTORY, exist_ok=True)
    print(f"所有输出将保存到目录: {BASE_SAVE_DIRECTORY}")

    try:
        print("\n--- 开始数据加载和预处理 ---")

        X_raw, y_raw, feature_names = load_and_preprocess_data(
            DATA_FILE_PATH, target_column=TARGET_COLUMN, exclude_columns=EXCLUDE_COLUMNS
        )

        if X_raw.shape[1] != INPUT_FEATURES_EXPECTED:
            raise ValueError(f"错误: 特征数量不匹配。预期 {INPUT_FEATURES_EXPECTED}，得到 {X_raw.shape[1]}。")

        print(f"\n正在将数据分割为主训练集 ({100*(1-TEST_SET_FRACTION)}%) 和最终测试集 ({100*TEST_SET_FRACTION}%) 进行无偏评估...")
        # **：在归一化之前进行第一次数据划分**
        X_main_train_raw, X_test_final_raw, y_main_train, y_test_final = train_test_split(
            X_raw, y_raw, test_size=TEST_SET_FRACTION, random_state=RANDOM_STATE
        )
        print(f"主训练集大小 (用于K-Fold): {X_main_train_raw.shape[0]} 个样本")
        print(f"最终测试集大小: {X_test_final_raw.shape[0]} 个样本")

        # **仅在主训练集上拟合 StandardScaler**
        feature_scaler = create_feature_scaler(X_main_train_raw)
        
        # 保存特征Scaler
        scaler_save_path = os.path.join(BASE_SAVE_DIRECTORY, 'feature_scaler.pkl')
        joblib.dump(feature_scaler, scaler_save_path)
        print(f"特征Scaler已保存到 {scaler_save_path}")

        # **使用拟合好的 StandardScaler 转换所有数据集**
        X_main_train_scaled = transform_features(X_main_train_raw, feature_scaler)
        X_test_final_scaled = transform_features(X_test_final_raw, feature_scaler)

        X_main_train_tensor = torch.tensor(X_main_train_scaled, dtype=torch.float32).to(device)
        y_main_train_tensor = torch.tensor(y_main_train, dtype=torch.float32).to(device)
        
        X_test_final_tensor = torch.tensor(X_test_final_scaled, dtype=torch.float32).to(device)
        y_test_final_np = y_test_final.flatten()

        print(f"\n--- 使用固定超参数开始K-Fold交叉验证 ---")
        print(f"超参数: 学习率={FIXED_LEARNING_RATE}, 批量大小={FIXED_BATCH_SIZE}, "
              f"Dropout率={FIXED_DROPOUT_RATE}, Epochs={FIXED_EPOCHS}")
        
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        all_fold_train_losses = []
        all_fold_val_losses = []
        all_fold_train_rmses = []
        all_fold_val_rmses = []
        
        best_overall_model_state = None # 用于存储来自最佳折叠的状态
        best_fold_number = -1
        best_val_rmse_across_folds = float('inf')

        for fold, (train_index, val_index) in enumerate(kf.split(X_main_train_tensor)):
            print(f"\n--- 训练折叠 {fold+1}/{K_FOLDS} ---")

            # **X_main_train_tensor 已经是归一化后的数据，所以 K-Fold 内部直接使用即可**
            X_train_fold, X_val_fold = X_main_train_tensor[train_index], X_main_train_tensor[val_index]
            y_train_fold, y_val_fold = y_main_train_tensor[train_index], y_main_train_tensor[val_index]

            model = create_regression_nn(INPUT_FEATURES_EXPECTED, dropout_rate=FIXED_DROPOUT_RATE)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=FIXED_LEARNING_RATE)
            
            fold_train_losses, fold_val_losses, fold_train_rmses, fold_val_rmses, current_fold_best_state, current_fold_best_rmse = \
                run_single_fold_training_cycle(
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                    model, criterion, optimizer, FIXED_EPOCHS, FIXED_BATCH_SIZE, 
                    BASE_SAVE_DIRECTORY, fold+1 # 直接传递主目录和折叠号
                )
            
            all_fold_train_losses.append(fold_train_losses)
            all_fold_val_losses.append(fold_val_losses)
            all_fold_train_rmses.append(fold_train_rmses)
            all_fold_val_rmses.append(fold_val_rmses)

            if current_fold_best_rmse < best_val_rmse_across_folds:
                best_val_rmse_across_folds = current_fold_best_rmse
                best_overall_model_state = current_fold_best_state
                best_fold_number = fold + 1
            
            print(f"折叠 {fold+1} 完成。此折叠的最佳验证RMSE: {current_fold_best_rmse:.4f}")

        print("\n--- K-Fold交叉验证完成 ---")
        print(f"在折叠 {best_fold_number} 中找到最佳模型状态，其验证RMSE为: {best_val_rmse_across_folds:.4f}")

        # 绘制K-Fold的平均损失和RMSE曲线
        # 确保所有列表长度相同以便求平均
        min_len = min(len(l) for l in all_fold_train_losses)
        avg_train_losses = np.mean([l[:min_len] for l in all_fold_train_losses], axis=0)
        avg_val_losses = np.mean([l[:min_len] for l in all_fold_val_losses], axis=0)
        avg_train_rmses = np.mean([l[:min_len] for l in all_fold_train_rmses], axis=0)
        avg_val_rmses = np.mean([l[:min_len] for l in all_fold_val_rmses], axis=0)

        plot_loss_and_rmse_curves(
            avg_train_losses, avg_train_rmses, avg_val_losses, avg_val_rmses,
            epochs_to_plot=min_len, # 绘制到最小公共长度
            save_path=os.path.join(BASE_SAVE_DIRECTORY, "average_kfold_curves.png"),
            title_prefix="Average K-Fold"
        )
        
        # 在最终测试集上评估最佳模型
        if best_overall_model_state is not None:
            final_model_for_test_eval = create_regression_nn(INPUT_FEATURES_EXPECTED, dropout_rate=FIXED_DROPOUT_RATE)
            final_model_for_test_eval.load_state_dict(best_overall_model_state)
            print("\n--- 正在最终保留测试集上评估最佳模型 ---")

            final_test_predictions_np, final_test_rmse, final_test_r2 = evaluate_model_on_test_set(
                final_model_for_test_eval, X_test_final_tensor, y_test_final_np
            )

            print(f"\n最终模型在测试集上的RMSE: {final_test_rmse:.4f}")
            print(f"最终模型在测试集上的R^2: {final_test_r2:.4f}")

            # 绘制DFT值与ML预测（最终测试集）图
            plot_dft_vs_ml(
                y_test_final_np, final_test_predictions_np,
                'DFT Value vs. ML Prediction (on Final Test Set)',
                os.path.join(BASE_SAVE_DIRECTORY, "dft_vs_ml_predictions_final_test_set.png")
            )

            # 保存最佳训练模型的状态（权重）
            final_model_state_path = os.path.join(BASE_SAVE_DIRECTORY, "best_trained_model_weights.pth")
            torch.save(final_model_for_test_eval.state_dict(), final_model_state_path)
            print(f"最佳训练模型权重 (来自折叠 {best_fold_number}) 已保存到: {final_model_state_path}")
            
            # 将最终结果保存到文件
            final_run_results = {
                "used_hyperparameters": {
                    "learning_rate": FIXED_LEARNING_RATE,
                    "batch_size": FIXED_BATCH_SIZE,
                    "dropout_rate": FIXED_DROPOUT_RATE,
                    "epochs": FIXED_EPOCHS
                },
                "best_kfold_validation_rmse": best_val_rmse_across_folds,
                "final_test_rmse": final_test_rmse,
                "final_test_r2": final_test_r2
            }
            final_results_path = os.path.join(BASE_SAVE_DIRECTORY, 'final_run_results.json')
            with open(final_results_path, 'w') as f:
                json.dump(final_run_results, f, indent=4)
            print(f"最终运行结果已保存到 {final_results_path}")

        else:
            print("错误: K-Fold交叉验证后未找到用于最终评估的最佳模型状态。")

        print(f"\n所有操作已完成！请检查 '{BASE_SAVE_DIRECTORY}' 文件夹以获取所有结果。")

    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
        print("请确保所有指定的数据文件都存在于正确的路径中。")
    except ValueError as e:
        print(f"数据处理错误: {e}")
    except Exception as e:
        print(f"发生意外错误: {e}")
        import traceback
        traceback.print_exc() # 打印意外错误的完整堆栈跟踪。