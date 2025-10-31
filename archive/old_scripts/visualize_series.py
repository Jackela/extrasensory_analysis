import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os

# Define file paths
PKL_FILE = 'database.pkl'
OUTPUT_IMAGE = 'timeseries_plot.png'
DOCS_FILE = 'PROJECT_DOCS.md'

def create_or_update_docs(dataset_name, data_shape, label):
    """Creates or updates the project documentation file."""
    content = f"""
# PROJECT_DOCS.md

## 1. 项目概述 (Project Overview)
本项目旨在分析和处理一个包含1053个多变量时间序列的数据集。该数据集来源于LMTS，并存储在 `database.pkl` 文件中。

## 2. 项目结构 (Project Structure)
```mermaid
graph TD
    A[database.pkl] --> B[Python Scripts];
    B --> C[Analysis & Visualization];
    C --> D[Models];
```

## 3. 核心组件与逻辑 (Core Components & Logic)
### 数据结构
- **`database.pkl`**: 包含一个Python字典。
- **顶层字典**: `{{ 'dataset_name': <inner_dict> }}`
- **内层字典**: `{{{{ 'data': <numpy.ndarray>, 'labels': <string> }}}}`

### 当前分析样本
- **数据集名称**: `{dataset_name}`
- **数据形状 (变量数, 时间点数)**: `{data_shape}`
- **标签**: `{label}`

## 4. 交互与数据流 (Interaction & Data Flow)
```mermaid
graph TD
    subgraph Data Loading
        A[Load database.pkl] --> B[Extract a sample dataset];
    end
    subgraph Visualization
        B --> C[Plot time series data];
        C --> D[Save plot as timeseries_plot.png];
    end
```
"""
    # Check if file exists and if content is different
    # Read with utf-8 encoding
    if not os.path.exists(DOCS_FILE) or open(DOCS_FILE, 'r', encoding='utf-8').read() != content:
        with open(DOCS_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"项目文档 '{DOCS_FILE}' 已创建/更新。")


def visualize_first_series():
    """Loads the first time series from the pkl file and generates a plot."""
    try:
        with open(PKL_FILE, 'rb') as f:
            data = pickle.load(f)

        # Get the first dataset
        first_key = list(data.keys())[0]
        time_series_data = data[first_key]['data']
        label = data[first_key]['labels']

        # --- Update Documentation ---
        create_or_update_docs(first_key, time_series_data.shape, label)

        # --- Visualization ---
        print(f"正在可视化第一个数据集: '{first_key}'...")
        print(f"数据形状 (变量数, 时间点数): {time_series_data.shape}")
        
        num_variables, num_time_points = time_series_data.shape
        
        plt.figure(figsize=(15, 8))
        
        # Plot each variable
        for i in range(num_variables):
            plt.plot(time_series_data[i, :], label=f'Variable {i+1}')
            
        plt.title(f'Time Series Plot for: {first_key}')
        plt.xlabel('Time Points')
        plt.ylabel('Value')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(OUTPUT_IMAGE)
        print(f"图表已保存为 '{OUTPUT_IMAGE}'。请在您的本地文件系统中查看。")

    except FileNotFoundError:
        print(f"错误: 未找到 '{PKL_FILE}'。")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    visualize_first_series()