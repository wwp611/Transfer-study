# main_gbrt_script.py
import numpy as np
import os
import joblib # 用于保存和加载 scaler 和 GBRT 模型

from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt # 如果要生成 SHAP 图，仍然需要
import shap # 导入 shap 用于 GBRT 解释

# 从您的工具文件中导入函数
from data_utils import load_and_preprocess_data, create_feature_scaler, transform_features
from plot_utils import plot_dft_vs_ml, plot_learning_curve, plot_target_distribution

# --- 配置设备 (GBRT 主要在 CPU 上运行，无需显式 CUDA 配置) ---
print("正在使用 CPU 进行 GBRT 训练和预测。")

# --- GBRT 模型创建函数 (使用最佳参数) ---
def create_gbrt_model(learning_rate=0.1, max_depth=5, n_estimators=300, subsample=0.7, random_state=42):
    """
    创建并返回一个用于回归任务的梯度提升回归树模型，使用最佳参数。
    """
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=random_state
    )
    print(f"GBRT 模型已创建，参数为: n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}, subsample={subsample}。")
    return model

# --- 训练和评估模型的主函数 (修改为 GBRT，并添加学习曲线数据收集) ---
def train_and_evaluate_model_gbrt(X_train_main, y_train_main, k_folds=10,
                                  gbrt_params={'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.7},
                                  model_save_dir="GBRT_models_tuned"):
    """
    执行带交叉验证的 GBRT 模型训练和评估。
    收集数据以绘制学习曲线。
    此函数现在在 `X_train_main` 上进行 K-Fold。
    """
    y_train_main = y_train_main.flatten() # GBRT 通常期望一维目标

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    all_fold_val_rmses = []

    # 用于学习曲线的数据
    all_train_errors = []
    all_val_errors = []

    fold_val_indices = {} # 存储每个折叠的验证集索引 (相对于 X_train_main)

    best_fold_model = None # 存储验证 RMSE 最低的那个模型的实例
    best_val_rmse = float('inf')

    os.makedirs(model_save_dir, exist_ok=True)
    print(f"\n正在对训练数据执行 {k_folds} 折交叉验证，使用调整后的 GBRT 参数...")

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_main)):
        print(f"\n--- 折叠 {fold+1}/{k_folds} ---")
        fold_val_indices[fold + 1] = val_index # 记录验证集索引

        X_train, X_val = X_train_main[train_index], X_train_main[val_index]
        y_train, y_val = y_train_main[train_index], y_train_main[val_index]

        model = create_gbrt_model(**gbrt_params) # 使用解包字典传入参数

        # 训练模型
        model.fit(X_train, y_train)
        print(f"折叠 {fold+1} 的 GBRT 模型已训练。")

        # 评估
        val_preds = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        val_r2 = r2_score(y_val, val_preds)

        print(f"折叠 {fold+1} 验证 RMSE: {val_rmse:.4f}, 验证 R2: {val_r2:.4f}")
        all_fold_val_rmses.append(val_rmse)

        # 检查并保存当前最佳模型
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_fold_model = model # 保存整个模型实例
            print(f"     折叠 {fold+1} 具有目前最佳的验证 RMSE: {best_val_rmse:.4f}。模型已存储。")

        # 收集学习曲线数据 (逐棵树的误差)
        train_errors = []
        val_errors = []
        for i, train_pred in enumerate(model.staged_predict(X_train)):
            train_errors.append(np.sqrt(mean_squared_error(y_train, train_pred)))
        for i, val_pred in enumerate(model.staged_predict(X_val)):
            val_errors.append(np.sqrt(mean_squared_error(y_val, val_pred)))

        all_train_errors.append(train_errors)
        all_val_errors.append(val_errors)

    print("\n交叉验证完成。")
    print(f"所有折叠的平均验证 RMSE: {np.mean(all_fold_val_rmses):.4f}")

    # --- 绘制学习曲线 (基于平均误差) ---
    min_len = min(len(e) for e in all_train_errors)
    avg_train_rmses_curve = np.mean([e[:min_len] for e in all_train_errors], axis=0)
    avg_val_rmses_curve = np.mean([e[:min_len] for e in all_val_errors], axis=0)

    plot_learning_curve(
        avg_train_rmses_curve,
        avg_val_rmses_curve,
        n_estimators=min_len, # 达到的最大估计器数量
        save_path=os.path.join(model_save_dir, "gbrt_learning_curve.png"),
        model_name="GBRT"
    )

    return {
        'val_rmses': all_fold_val_rmses,
        'fold_val_indices': fold_val_indices,
        'best_gbrt_model': best_fold_model # 返回最佳模型实例
    }

# --- 主执行块 ---
if __name__ == "__main__":
    # --- 配置您的数据路径和列信息 (已更新为 WSL 兼容路径) ---
    data_file_path = "/mnt/c/Users/18769/Desktop/Transform Learning/data/formation_data.csv"
    target_col = 'target'
    exclude_cols = ['formula']
    INPUT_FEATURES_EXPECTED = 132 # 预期特征数量

    # --- 配置 GBRT 最佳参数 ---
    best_gbrt_params = {
        'learning_rate': 0.05,
        'max_depth': 5,
        'n_estimators': 300,
        'subsample': 0.7,
        'random_state': 42 # 保持随机种子以确保可复现性
    }

    TEST_SET_FRACTION = 0.2 # 用于最终测试集的数据比例（例如，0.2表示20%）
    RANDOM_STATE_SPLIT = 42 # 用于数据划分的随机种子

    save_directory = "/mnt/c/Users/18769/Desktop/Transform Learning/GBRT4formation" # 更改保存目录名
    os.makedirs(save_directory, exist_ok=True) # 确保主保存目录存在

    try:
        # 1. 加载和预处理原始数据
        X_raw, y_raw, feature_names = load_and_preprocess_data(
            data_file_path, target_column=target_col, exclude_columns=exclude_cols
        )

        if X_raw.shape[1] != INPUT_FEATURES_EXPECTED:
            print(f"错误: 经过预处理后特征数量不为{INPUT_FEATURES_EXPECTED}，请检查数据或预处理逻辑。实际特征数: {X_raw.shape[1]}")
            exit()

        # --- 在归一化之前进行训练集和测试集的划分 ---
        print(f"\n正在将数据分割为主训练集 ({100*(1-TEST_SET_FRACTION)}%) 和最终测试集 ({100*TEST_SET_FRACTION}%) 进行无偏评估...")
        X_main_train_raw, X_test_final_raw, y_main_train, y_test_final = train_test_split(
            X_raw, y_raw, test_size=TEST_SET_FRACTION, random_state=RANDOM_STATE_SPLIT
        )
        print(f"主训练集大小 (用于 K-Fold): {X_main_train_raw.shape[0]} 个样本")
        print(f"最终测试集大小: {X_test_final_raw.shape[0]} 个样本")

        # --- 仅在主训练集上拟合 StandardScaler ---
        feature_scaler = create_feature_scaler(X_main_train_raw)

        # 保存特征 Scaler
        scaler_save_path = os.path.join(save_directory, 'feature_scaler.pkl')
        joblib.dump(feature_scaler, scaler_save_path)
        print(f"特征 Scaler 已保存到 {scaler_save_path}")

        # --- 使用拟合好的 StandardScaler 转换所有数据集 ---
        X_main_train_normalized = transform_features(X_main_train_raw, feature_scaler)
        X_test_final_normalized = transform_features(X_test_final_raw, feature_scaler)

        print(f"\n--- 正在使用最佳参数开始 GBRT 训练: ---")
        for param, value in best_gbrt_params.items():
            print(f"   {param}: {value}")

        # 3. 训练和评估 GBRT 模型 (在归一化后的主训练集上进行 K 折交叉验证)
        results_summary = train_and_evaluate_model_gbrt(
            X_train_main=X_main_train_normalized, # 传入归一化后的主训练集
            y_train_main=y_main_train, # 传入主训练集的目标
            k_folds=10,
            gbrt_params=best_gbrt_params,
            model_save_dir=save_directory
        )

        avg_val_rmse_across_folds = np.mean(results_summary['val_rmses'])
        print(f"\n所有 GBRT 折叠的平均验证 RMSE (使用最佳参数): {avg_val_rmse_across_folds:.4f}")

        best_gbrt_model = results_summary['best_gbrt_model']

        # --- 在最终测试集上评估最佳模型并绘图 ---
        if best_gbrt_model:
            print("\n--- 正在最终测试集上评估最佳 GBRT 模型 ---")
            y_test_final_flat = y_test_final.flatten() # 确保 y_test_final 也是一维的

            test_predictions = best_gbrt_model.predict(X_test_final_normalized)
            test_rmse = np.sqrt(mean_squared_error(y_test_final_flat, test_predictions))
            test_r2 = r2_score(y_test_final_flat, test_predictions)

            print(f"最终测试集 RMSE: {test_rmse:.4f}")
            print(f"最终测试集 R2: {test_r2:.4f}")

            # 绘制最终测试集的 DFT vs ML 预测图
            plot_dft_vs_ml(
                y_test_final_flat, test_predictions,
                'DFT value vs. GBRT predictions (on final test set)',
                os.path.join(save_directory, "dft_vs_gbrt_predictions_final_test_set.png")
            )

            # 保存最终使用的最佳 GBRT 模型
            final_model_path = os.path.join(save_directory, "best_gbrt_model_for_test.joblib")
            joblib.dump(best_gbrt_model, final_model_path)
            print(f"最佳 GBRT 模型已保存到 {final_model_path}")
        else:
            print("警告: 未从 K 折交叉验证中找到最佳 GBRT 模型。无法在测试集上进行评估。")

        print("\n所有操作完成！GBRT 模型、学习曲线、预测图和特征归一化器已保存。")

        # --- SHAP 分析部分 (针对 GBRT) ---
        print("\n--- 正在开始 GBRT 的 SHAP 分析 (使用 TreeExplainer) ---")
        if best_gbrt_model: # 使用在整个训练集上找到的最佳模型进行 SHAP 分析
            explainer = shap.TreeExplainer(best_gbrt_model)
            # 对整个主训练集（归一化后）进行 SHAP 分析
            print(f"正在计算 {X_main_train_normalized.shape[0]} 个样本的 SHAP 值...")
            shap_values = explainer.shap_values(X_main_train_normalized)
            print("SHAP 值计算完成。")

            mean_abs_shap_values = np.abs(shap_values).mean(axis=0)

            import pandas as pd # 确保 pandas 已导入
            feature_importance = pd.DataFrame({
                'Feature': feature_names, # 特征名称是基于原始 X_raw 的
                'SHAP_Importance': mean_abs_shap_values
            })

            feature_importance = feature_importance.sort_values(by='SHAP_Importance', ascending=False)

            print("\nGBRT 特征重要性 (前 15):")
            print(feature_importance.head(15).to_string())

            shap.summary_plot(shap_values, X_main_train_normalized, feature_names=feature_names,
                             plot_type="bar", max_display=15, show=False)
            plt.title('SHAP  (GBRT Top 15 Features)')
            gbrt_shap_bar_plot_path = os.path.join(save_directory, "gbrt_shap_bar_plot_top15.png")
            plt.savefig(gbrt_shap_bar_plot_path, bbox_inches='tight')
            print(f"GBRT SHAP bar plot (Top 15) saved to: {gbrt_shap_bar_plot_path}")
            plt.close()

            shap.summary_plot(shap_values, X_main_train_normalized, feature_names=feature_names,
                             max_display=15, show=False)
            plt.title('SHAP summary plot (GBRT Top 15 Features)')
            gbrt_shap_summary_plot_path = os.path.join(save_directory, "gbrt_shap_summary_plot_top15.png")
            plt.savefig(gbrt_shap_summary_plot_path, bbox_inches='tight')
            print(f"GBRT SHAP summary plot (Top 15) saved to: {gbrt_shap_summary_plot_path}")
            plt.close()
            print("GBRT SHAP analysis completed!")
        else:
            print("警告: 最佳训练的 GBRT 模型不可用于 SHAP 分析。")


        # --- 数据分布分析部分 (使用 plot_utils 中的函数) ---
        print("\n--- 正在分析折叠 4、8 和 5 的数据分布 ---")

        # 注意：这里 y_raw 是原始的全量目标值，fold_val_indices 是相对于 X_main_train_normalized 的索引
        # 所以我们需要用这些索引从 y_main_train 中获取对应的目标值
        y_main_train_flat = y_main_train.flatten() # 展平主训练集的目标值

        if 'fold_val_indices' in results_summary and \
           4 in results_summary['fold_val_indices'] and \
           8 in results_summary['fold_val_indices'] and \
           5 in results_summary['fold_val_indices']:

            fold_4_val_indices_in_main_train = results_summary['fold_val_indices'][4]
            fold_8_val_indices_in_main_train = results_summary['fold_val_indices'][8]
            fold_5_val_indices_in_main_train = results_summary['fold_val_indices'][5]

            fold_targets_for_plot = {
                'fold 4': y_main_train_flat[fold_4_val_indices_in_main_train],
                'fold 8': y_main_train_flat[fold_8_val_indices_in_main_train],
                'fold 5': y_main_train_flat[fold_5_val_indices_in_main_train]
            }

            plot_target_distribution(fold_targets_for_plot, os.path.join(save_directory, "folds_target_distribution.png"))

        else:
            print("警告: 无法对折叠 4、8、5 执行数据分布分析。'fold_val_indices' 不可用或不完整。")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()