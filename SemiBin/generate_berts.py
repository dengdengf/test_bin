import argparse
import time
from pathlib import Path
import torch
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.decomposition import PCA

# ----------------------------------------------------------------
# 参数设置
# ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="基于DNABERT提取特征并降维 (全量提取，无长度过滤)")
parser.add_argument('-md', '--model_dir', type=str, required=True, help="指定模型路径 (包含 config.json, pytorch_model.bin 等)")
parser.add_argument('-fd', '--fasta_file', type=str, required=True, help="指定输入 FASTA 序列路径")
parser.add_argument('-nd', '--names_file', type=str, required=True, help="输出的 contig 名称文件路径 (.txt)")
parser.add_argument('-dd', '--fpf_file', type=str, required=True, help="输出的特征文件路径 (.npy)")
args = parser.parse_args()

# ----------------------------------------------------------------
# 初始化环境与模型
# ----------------------------------------------------------------
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 路径处理
model_dir = Path(args.model_dir).resolve()

print(f"正在加载模型: {model_dir} ...")
try:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True).to(device)
except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)

# ----------------------------------------------------------------
# Step 1: 读取序列 (直接读入内存，不产生中间文件)
# ----------------------------------------------------------------
seq_list = []       # 存放序列内容
contig_names = []   # 存放序列名称

print(f"正在读取 FASTA 文件: {args.fasta_file} ...")
# 使用 SeqIO 解析，确保名称和序列顺序严格一致
for record in tqdm(SeqIO.parse(args.fasta_file, "fasta"), desc="读取进度"):
    # 1. 保存名称 (record.id)
    contig_names.append(record.id)
    # 2. 保存序列 (转为字符串)
    seq_list.append(str(record.seq))

print(f"共读取到 {len(seq_list)} 条序列。")

if len(seq_list) == 0:
    print("错误：输入文件没有包含任何序列。")
    exit(1)

# ----------------------------------------------------------------
# Step 2: 提取特征
# ----------------------------------------------------------------
feature_list = []
start_time = time.time()

print("开始提取特征...")
model.eval() # 切换到评估模式

# 遍历内存中的序列列表
for seq in tqdm(seq_list, desc="特征提取中"):
    # Tokenize
    try:
        inputs = tokenizer(seq, 
                        return_tensors='pt',
                        padding="longest",
                        max_length=5000,
                        truncation=True)["input_ids"]
        inputs = inputs.to(device)
        
        # Forward pass
        with torch.no_grad():
            hidden_states = model(inputs)[0] # [1, sequence_length, 768]
            # Mean pooling: 对序列维度的特征取平均
            embedding_mean = torch.mean(hidden_states[0], dim=0)
            feature_list.append(embedding_mean.detach().cpu().numpy())
            
    except Exception as e:
        print(f"\n处理序列时发生错误: {e}")
        # 即使出错也要保持对齐，可以用全0向量填充，或者直接报错退出。
        # 这里选择报错退出以保证数据严谨性
        exit(1)

# ----------------------------------------------------------------
# Step 3: PCA 降维
# ----------------------------------------------------------------
features_array = np.stack(feature_list)
print(f"\n原始特征矩阵形状: {features_array.shape}")

# 动态调整 PCA 维度（防止序列数少于 128 时报错）
target_dim = 128
n_samples = features_array.shape[0]
n_components = min(target_dim, n_samples)

if n_components < target_dim:
    print(f"警告：样本数量 ({n_samples}) 少于目标维度 128，PCA 维度自动调整为 {n_components}")

print(f"正在进行 PCA 降维 (目标维度: {n_components})...")
pca = PCA(n_components=n_components)
reduced_features = pca.fit_transform(features_array)

print(f"降维后特征矩阵形状: {reduced_features.shape}")
print(f"PCA 解释方差比 (Total): {np.sum(pca.explained_variance_ratio_):.4f}")

# ----------------------------------------------------------------
# Step 4: 保存结果
# ----------------------------------------------------------------
print("正在保存文件...")

# 1. 保存特征 (.npy)
np.save(args.fpf_file, reduced_features)

# 2. 保存名称列表 (.txt)
# 确保顺序与 feature_list 也就是 reduced_features 严格对应
with open(args.names_file, 'w', encoding='utf-8') as f:
    for name in contig_names:
        f.write(f"{name}\n")

end_time = time.time()
print("-" * 30)
print(f"处理完成！耗时: {end_time - start_time:.2f} 秒")
print(f"特征文件已保存至: {args.fpf_file}")
print(f"名称列表已保存至: {args.names_file}")
print(f"两条数检查: 名称数={len(contig_names)}, 特征数={len(reduced_features)}")
















# import argparse
# import time
# from pathlib import Path

# import torch
# import numpy as np
# import pandas as pd
# from Bio import SeqIO
# from transformers import AutoTokenizer, AutoModel
# from tqdm import tqdm
# # !!! 新增的导入: 用于降维 !!!
# from sklearn.decomposition import PCA 


# # 设置参数
# parser = argparse.ArgumentParser(description="用于特征提取:model_dir、seq.txt、feature_output.npy")
# parser.add_argument('-md','--model_dir',type=str,help="指定模型路径")
# parser.add_argument('-fd','--fasta_file',type=str,help="指定输入序列路径")
# parser.add_argument('-sd','--seq_file',type=str,help="提取序列路径")
# parser.add_argument('-cf', '--contig_names_file', type=str, help="提取的contig名称路径")
# parser.add_argument('-dd','--fpf_file',type=str,help="指定特征输出路径")
# parser.add_argument('--csv_file', type=str, help="指定CSV输出路径") # 提示用户此处是CSV
# args = parser.parse_args()

# # 设置设备为GPU（如果可用）
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # 统一解析模型目录为绝对路径（支持相对路径）
# model_dir = Path(args.model_dir).resolve()

# tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
# model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True).to(device)

# # Step 1: 筛选长度大于等于2000的序列，并保存名字和序列
# contig_names_list_temp = []
# with open(args.seq_file, "w") as seq_output_file, open(args.contig_names_file, "w") as contig_names_output_file:
#     for record in tqdm(SeqIO.parse(args.fasta_file, "fasta")):
#         seq = str(record.seq)
#         if len(seq) >= 2000:
#             # 保存序列
#             seq_output_file.write(f">{record.id}\n{seq}\n")
#             # 保存contig名称（序列的标识符）
#             contig_names_output_file.write(record.id + '\n')  # record.id 是序列名称
#             contig_names_list_temp.append(record.id) # 提前收集contig name，避免重复读取

# time.sleep(5)


# # Step 2 & 3: 读取序列文件和contig_names文件
# # 读取序列文件（通常包含FASTA格式的头和序列行）
# with open(args.seq_file, "r", encoding="UTF-8") as f:
#     # 过滤掉FASTA头行，只保留序列行
#     seq_lines = [line for line in f.read().splitlines() if not line.startswith(">")]

# # 读取contig_names文件
# with open(args.contig_names_file, "r", encoding="UTF-8") as f:
#     contig_names = f.read().splitlines()


# feature_list = []
# start_time = time.time()
# for seq in tqdm(seq_lines,desc = "正在提取特征......"):
#     inputs = tokenizer(seq, 
#                        return_tensors = 'pt',
#                        padding="longest",
#                        max_length=5000,
#                        truncation=True,)["input_ids"]
#     inputs = inputs.to(device)  # 将输入数据移动到GPU
#     with torch.no_grad():  # 禁用梯度计算以节省内存
#         hidden_states = model(inputs)[0] # [1, sequence_length, 768]
#         # embedding with mean pooling
#         embedding_mean = torch.mean(hidden_states[0], dim=0)
#         feature_list.append(embedding_mean.detach().cpu().numpy())


# # ----------------------------------------------------
# # 步骤 4: 降维处理 (保持不变)
# # ----------------------------------------------------
# # 将特征列表转换为 NumPy 数组 (形状: N x 768)
# features_array = np.stack(feature_list)

# # 检查 contig_names 和特征数量是否匹配
# assert len(contig_names) == len(features_array), "contig_names 和特征数量不匹配！"

# print("\n--- 降维信息 ---")
# print(f"原始特征维度: {features_array.shape}")

# # 初始化 PCA，目标维度设置为 128
# pca = PCA(n_components=128)

# # 对特征进行拟合和转换，得到降维后的特征
# reduced_features = pca.fit_transform(features_array)

# print(f"降维后的特征维度: {reduced_features.shape}")
# print(f"PCA 解释的方差比例 (前128维): {np.sum(pca.explained_variance_ratio_):.4f}")
# print("----------------")


# # ----------------------------------------------------
# # 步骤 5: 保存降维后的特征 (已修改为 CSV)
# # ----------------------------------------------------

# # 使用降维后的特征进行保存 (.npy 文件)
# np.save(args.fpf_file, reduced_features)

# # 转换为 pandas DataFrame
# features_df = pd.DataFrame(reduced_features)

# # 将 contig_name 列添加到特征中，保证与序列一一对应
# features_df['contig_name'] = contig_names  

# # !!! 使用 .to_csv() 保存为 CSV 格式 !!!
# features_df.to_csv(args.csv_file, index=False)

# end_time = time.time()
# print("提取和降维特征耗费时间：%d 秒" %(end_time-start_time))
# print(f"PCA 解释的方差比例 (前128维): {np.sum(pca.explained_variance_ratio_):.4f}")
# print(f"成功保存 {len(feature_list)} 条序列的 128 维特征到 CSV 和 NumPy 文件。")