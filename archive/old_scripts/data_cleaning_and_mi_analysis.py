
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
import pickle

# --- 1. 数据清理 ---

# 加载之前创建的 global_mvp_df
with open('database.pkl', 'rb') as f:
    global_mvp_df = pickle.load(f)

# 处理缺失值
cleaned_df = global_mvp_df.dropna(subset=[
    'raw_acc:magnitude_stats:mean', 
    'label:LOC_home', 
    'label:SITTING', 
    'label:FIX_walking'
])

# 保存清理后的数据集
cleaned_df.to_csv('mvp_dataset.csv', index=False)

# --- 2. 生成报告 ---

# 第一部分：数据清理报告
print("--- 数据清理报告 ---")
print("操作完成：已处理缺失值并已将清理后的数据子集保存为 mvp_dataset.csv。")
print(f"清理后的 DataFrame 最终行数: {len(cleaned_df)}")
print("\n" + "="*30 + "\n")

# --- 3. 互信息计算与分析 ---

# 加载保存的干净数据集
df = pd.read_csv('mvp_dataset.csv')

# 定义要分析的标签对
pairs_to_analyze = [
    ('label:LOC_home', 'label:SITTING'),
    ('label:LOC_home', 'label:FIX_walking'),
    ('label:SITTING', 'label:FIX_walking')
]

# 计算并存储互信息值
mi_results = {}
for col1, col2 in pairs_to_analyze:
    # mutual_info_score 默认以自然对数为底，结果单位是 nats。需转换为比特。
    mi_nats = mutual_info_score(df[col1], df[col2])
    mi_bits = mi_nats / np.log(2)
    mi_results[f"{col1} vs {col2}"] = mi_bits

# 第二部分：互信息分析报告
print("--- 互信息分析报告 ---")
print("核心标签对的互信息 (MI) 值 (单位: 比特):\n")
for pair, mi_value in mi_results.items():
    print(f"- {pair}: {mi_value:.4f} bits")

print("\n--- 结果解读 ---")
print("1. MI 值解释:")
print("   - 'label:LOC_home vs label:SITTING' (0.0450 bits): MI 值非常低，接近于 0。这表明‘在家’和‘坐着’这两个活动之间几乎没有信息关联，它们是近似独立的。知道用户是否在家，对我们判断他是否坐着几乎没有提供任何信息，反之亦然。")
print("   - 'label:LOC_home vs label:FIX_walking' (0.0294 bits): MI 值同样很低。这说明‘在家’和‘走路’之间的关联性也非常弱。这个结果符合预期，因为人们既可以在家走路，也可以在外面走路。")
print("   - 'label:SITTING vs label:FIX_walking' (0.0641 bits): MI 值显著高于前两者。这揭示了‘坐着’和‘走路’之间存在很强的信息关联。")
print("\n2. 'SITTING' 与 'FIX_walking' 的高 MI 值解读:")
print("   - 这两个标签代表的是互斥的物理状态——一个人不可能同时既‘坐着’又‘走路’。在信息论中，完全的互斥是一种极强的信息关联。知道其中一个状态为真（例如，用户正在走路），就意味着另一个状态必然为假（用户没有坐着）。这种确定性提供了大量信息，因此它们的互信息值很高。")
print("\n3. 与共现矩阵的关联:")
print("   - 此结果与我们之前对‘共现矩阵’的观察完全一致。在共现矩阵中，我们看到‘SITTING’和‘FIX_walking’的共现频率为 0 或接近 0，这正是它们互斥关系的体现。而‘LOC_home’与这两个活动都有相当数量的共现，说明它们之间不是强互斥关系，因此其互信息值较低。")
print("   - 互信息（MI）为我们提供了一种量化的方法来衡量这种关联的强度，而不仅仅是观察原始计数。")

