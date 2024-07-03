import pandas as pd
import matplotlib.pyplot as plt

#数据清洗
def clean_data(data):
    #copy副本
    data_cleaned = data.copy()
    
    # 填充缺失值
    for col in data_cleaned.columns:
        if data_cleaned[col].dtype in ['float64', 'int64']:
            median_value = data_cleaned[col].median()
            data_cleaned[col].fillna(median_value, inplace=True)
    
    # 处理异常值
    for col in data_cleaned.columns:
        if data_cleaned[col].dtype in ['float64', 'int64']:
            Q1 = data_cleaned[col].quantile(0.25)
            Q3 = data_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data_cleaned = data_cleaned[(data_cleaned[col] >= lower_bound) & (data_cleaned[col] <= upper_bound)]
    
    return data_cleaned

#加载数据
def load_data(filepath):
    """加载并初步处理数据。"""
    data = pd.read_csv(filepath, encoding='utf-8')
    data.rename(columns=lambda x: x.strip(), inplace=True)
    return data

def visualize_data(data):
    """数据可视化。"""
    fig, axes = plt.subplots(3, 5, figsize=(15, 10))
    for i, col in enumerate(data.columns[:-1]):
        axes[i//5, i%5].boxplot(data[col])
        axes[i//5, i%5].set_title(col)
    plt.tight_layout()
    plt.show()
