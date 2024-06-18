import pandas as pd

# 文件路径
input_file = 'input.txt'
output_file = 'output.csv'

# 读取txt文件
data = []
with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        # 以空格为间隔分割每行数据
        data.append(line.strip().split())

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 将DataFrame写入csv文件
df.to_csv(output_file, index=False, header=False, encoding='utf-8')

print(f"数据已成功写入 {output_file}")