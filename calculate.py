import numpy as np

def calculate_mean_std(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return f"{mean:.2f} ± {std:.2f}"

# 示例数组
array = [82.56,82.70,83.89,82.12,82.97]

# 计算并显示结果
result = calculate_mean_std(array)
print(result)
