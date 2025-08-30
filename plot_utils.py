# plot_utils.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os

def plot_dft_vs_ml(true_values, predictions, title, save_path, annotate_metrics=True):
    """绘制 DFT (真实目标) vs. ML 预测。"""
    if not len(true_values) or not len(predictions):
        print(f"警告: 无法绘制 '{title}' 的 DFT vs. ML，因为真实值或预测值为空。")
        return

    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predictions, alpha=0.6, color='darkorange')

    min_val = min(true_values.min(), predictions.min())
    max_val = max(true_values.max(), predictions.max())
    # 根据实际数据范围调整绘图限制
    plot_min = min(min_val, predictions.min(), true_values.min()) - 0.5
    plot_max = max(max_val, predictions.max(), true_values.max()) + 0.5

    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', label='perfect prediction')

    plt.title(title)
    plt.xlabel('DFT values (real target)')
    plt.ylabel('ML predictions')
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
        print(f"'{title}' 绘图已保存到 {save_path}")
    except Exception as e:
        print(f"错误: 无法将 '{title}' 绘图保存到 {save_path}。原因: {e}")
    finally:
        plt.close()

def plot_learning_curve(avg_train_rmses_curve, avg_val_rmses_curve, n_estimators, save_path, model_name="Model"):
    """
    绘制学习曲线 (RMSE vs. 估计器数量)。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_estimators + 1), avg_train_rmses_curve, label='average RMSE', color='blue')
    plt.plot(range(1, n_estimators + 1), avg_val_rmses_curve, label='average validation RMSE', color='red')
    plt.title(f'{model_name} learning curve (RMSE vs. number of estimators)')
    plt.xlabel('Number of estimators (trees)')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend()
    try:
        plt.savefig(save_path)
        print(f"{model_name} 学习曲线已保存到 {save_path}")
    except Exception as e:
        print(f"错误: 无法将学习曲线图保存到 {save_path}。原因: {e}")
    finally:
        plt.close()

def plot_target_distribution(fold_targets_dict, save_path):
    """
    绘制指定折叠的验证集目标值分布。
    fold_targets_dict: 字典，键为折叠名称（如 'Fold 4'），值为对应的目标值数组。
    """
    num_plots = len(fold_targets_dict)
    if num_plots == 0:
        print("未提供折叠目标进行分布绘图。")
        return

    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6))
    if num_plots == 1: # 处理单个子图的情况
        axes = [axes]

    colors = ['orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan']
    
    for i, (fold_name, targets) in enumerate(fold_targets_dict.items()):
        ax = axes[i]
        ax.hist(targets, bins=30, density=True, alpha=0.7, color=colors[i % len(colors)], edgecolor='black')
        ax.set_title(f'{fold_name} distribution')
        ax.set_xlabel('Target values')
        ax.set_ylabel('Density')
        ax.grid(True, linestyle='--', alpha=0.6)

        print(f"\n{fold_name} validation target statistics:")
        print(f"   Min: {np.min(targets):.4f}, Max: {np.max(targets):.4f}")
        print(f"   Mean: {np.mean(targets):.4f}, Std: {np.std(targets):.4f}")
        print(f"   Median: {np.median(targets):.4f}")

    plt.tight_layout()
    try:
        plt.savefig(save_path)
        print(f"Target distribution plot saved to {save_path}")
    except Exception as e:
        print(f"Error: Unable to save distribution plot to {save_path}. Reason: {e}")
    finally:
        plt.close()
