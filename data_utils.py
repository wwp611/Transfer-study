# data_utils.py
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

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

    # 确保 exclude_columns 是一个列表
    if not isinstance(exclude_columns, list):
        exclude_columns = [exclude_columns]
        
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

    # 更改此打印语句以更具动态性
    removed_cols_str = ', '.join(cols_to_drop)
    print(f"列 '{removed_cols_str}' 已从特征中移除。")
    print(f"处理后的特征数量: {X_data.shape[1]}")

    return X_data, y_data, feature_names

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
