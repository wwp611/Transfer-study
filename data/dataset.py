import pandas as pd

# 输入文件
input_file = 'Original_data.xlsx'

# 1. 读取 Excel 文件
try:
    df = pd.read_excel(input_file, engine='openpyxl')
    print(f"成功读取 Excel 文件 {input_file}，包含 {len(df.columns)} 列")
except Exception as e:
    print(f"读取 Excel 文件失败: {e}")
    print("请确认以下事项：")
    print("1. 已安装 openpyxl（运行：pip install openpyxl）")
    print("2. Original_data.xlsx 是否为有效的 Excel 文件")
    print("3. 文件路径是否正确（当前路径：C:\\Users\\18769\\Desktop\\Transform Learning\\data\\）")
    print("请用 Excel 打开 Original_data.xlsx，复制前5行数据（包括列名）以供调试。")
    exit()

# 2. 检查列数和列名
print(f"数据包含 {len(df.columns)} 列，列名：{list(df.columns)}")

# 3. 定义目标列及其对应的输出文件名
targets = {
    'formation_energy_per_atom': 'formation_data.csv',
    'e_above_hull': 'Ehull_data.csv',
    'e_above_hull_class': 'Ehull_class_data.csv',
    'band_gap': 'bandgap_data.csv',
    'B/GPa': 'B.csv',
    'G/GPa': 'G.csv'
}

# 4. 确定特征列（排除 formula 和所有目标列）
target_columns = list(targets.keys())
all_columns = df.columns.tolist()
feature_columns = [col for col in all_columns if col not in ['formula'] + target_columns]
print(f"特征列数量：{len(feature_columns)}，特征列：{feature_columns}")

# 5. 拆分文件
for target, output_file in targets.items():
    if target in df.columns and 'formula' in df.columns:
        # 选择 formula、目标列（重命名为 target）和其他特征列
        # 排除其他目标列
        other_targets = [t for t in target_columns if t != target]
        columns_to_select = ['formula'] + [target] + [col for col in feature_columns if col not in other_targets]
        temp_df = df[columns_to_select].copy()
        
        # 重命名目标列为 target
        temp_df = temp_df.rename(columns={target: 'target'})
        
        # 对于 B 和 G 列，删除包含 NaN 的行
        if target in ['B/GPa', 'G/GPa']:
            temp_df = temp_df.dropna(subset=['target'])
            print(f"{output_file} 已去除 target 列（原 {target}）的空值行，剩余 {len(temp_df)} 行")
        
        # 保存到新的 CSV 文件
        temp_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"已生成 {output_file}，包含列：{list(temp_df.columns)}")
    else:
        print(f"警告：列 {target} 或 formula 不存在于文件中，跳过生成 {output_file}")

print("CSV文件拆分完成！")