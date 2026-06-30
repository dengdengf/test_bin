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
# 用法 1 (旧行为, 向后兼容): 只给 whole 的三个参数 -> 对该 fasta 提特征 + PCA 后保存。
# 用法 2 (方案 A, 推荐): 额外给 split 的三个参数 -> whole 与 split 一次跑完, 且 split 复用
#         whole 拟合出的同一个 PCA basis, 保证两者在同一坐标系 (对比学习才可比)。
#
# 注: split contig 的名称形如 h_1 / h_2 (见 generate_kmer.py), 它们不在原始 contig fasta 里,
#     所以 split fasta 需要是含这些半段序列的文件, 与 data_split.csv 的行顺序一致。
# (注释掉的旧版本已删除, 可用 `git show HEAD:SemiBin/generate_berts.py` 找回。)
# ----------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="基于DNABERT提取特征并降维 (全量提取, 无长度过滤; 支持 whole+split 共享 PCA basis)")
parser.add_argument('-md', '--model_dir', type=str, required=True,
                    help="指定模型路径 (包含 config.json, pytorch_model.bin 等)")
parser.add_argument('-fd', '--fasta_file', type=str, required=True,
                    help="whole contig 的 FASTA 路径 (与 data.csv 对应)")
parser.add_argument('-nd', '--names_file', type=str, required=True,
                    help="whole 的 contig 名称输出文件 (.txt)")
parser.add_argument('-dd', '--fpf_file', type=str, required=True,
                    help="whole 的特征输出文件 (.npy)")
# --- 可选: split contig 一并提取, 与 whole 共享同一个 PCA basis ---
parser.add_argument('-sfd', '--split_fasta_file', type=str, default=None,
                    help="(可选) split contig 的 FASTA 路径 (与 data_split.csv 对应, 名称形如 h_1/h_2)")
parser.add_argument('-snd', '--split_names_file', type=str, default=None,
                    help="(可选) split 的 contig 名称输出文件 (.txt)")
parser.add_argument('-sdd', '--split_fpf_file', type=str, default=None,
                    help="(可选) split 的特征输出文件 (.npy)")
# --- 可选: 性能 / 维度参数, 默认值保持原行为 ---
parser.add_argument('--batch_size', type=int, default=8,
                    help="批量推理 batch 大小 (默认 8; 设为 1 即逐条提取, 复现旧行为)")
parser.add_argument('--max_length', type=int, default=5000,
                    help="tokenizer 截断长度 (默认 5000)")
parser.add_argument('--target_dim', type=int, default=128,
                    help="PCA 目标维度 (默认 128)")
args = parser.parse_args()

if (args.split_fasta_file is not None) and (args.split_names_file is None or args.split_fpf_file is None):
    print("错误: 提供了 --split_fasta_file 时, 必须同时提供 --split_names_file 和 --split_fpf_file。")
    exit(1)

# ----------------------------------------------------------------
# 初始化环境与模型
# ----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_dir = Path(args.model_dir).resolve()
print(f"正在加载模型: {model_dir} ...")
try:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True).to(device)
except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)
model.eval()


def read_fasta(path):
    """读取 FASTA, 返回 (names, seqs), 顺序严格一致。"""
    names, seqs = [], []
    for record in tqdm(SeqIO.parse(path, "fasta"), desc=f"读取 {Path(path).name}"):
        names.append(record.id)
        seqs.append(str(record.seq))
    return names, seqs


def encode_sequences(seqs):
    """批量提取 DNABERT mean-pooling 特征。

    用 attention_mask 做掩码平均, 使 padding token 不污染嵌入 —— 这样批量(padding)的
    逐条结果与旧版 (batch=1, 无 padding) 在数值上保持一致, 只是更快。
    """
    if len(seqs) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    feature_list = []
    for start in tqdm(range(0, len(seqs), args.batch_size), desc="特征提取中"):
        batch = seqs[start:start + args.batch_size]
        enc = tokenizer(
            batch,
            return_tensors='pt',
            padding="longest",
            max_length=args.max_length,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            try:
                # 传入 attention_mask 让 padding 不参与注意力 (标准 BERT forward 支持)
                if attention_mask is not None:
                    hidden_states = model(input_ids, attention_mask=attention_mask)[0]
                else:
                    hidden_states = model(input_ids)[0]
            except TypeError:
                # 某些自定义 DNABERT 实现的 forward 不接受 attention_mask, 退回原始调用
                hidden_states = model(input_ids)[0]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            batch_emb = summed / counts
        else:
            batch_emb = hidden_states.mean(dim=1)
        feature_list.append(batch_emb.detach().cpu().numpy())
    return np.concatenate(feature_list, axis=0).astype(np.float32)


def save_names(path, names):
    with open(path, 'w', encoding='utf-8') as f:
        for name in names:
            f.write(f"{name}\n")


# ----------------------------------------------------------------
# Step 1+2: 读取并提取 whole contig 特征
# ----------------------------------------------------------------
start_time = time.time()
print(f"正在读取 whole FASTA: {args.fasta_file} ...")
whole_names, whole_seqs = read_fasta(args.fasta_file)
print(f"共读取到 {len(whole_seqs)} 条 whole 序列。")
if len(whole_seqs) == 0:
    print("错误: whole 输入文件没有包含任何序列。")
    exit(1)

whole_features = encode_sequences(whole_seqs)
print(f"\nwhole 原始特征矩阵形状: {whole_features.shape}")

# ----------------------------------------------------------------
# Step 3: 在 whole 上拟合 PCA (basis), 供 split 复用
# ----------------------------------------------------------------
target_dim = args.target_dim
n_samples, n_features = whole_features.shape
n_components = min(target_dim, n_samples, n_features)

reducer = None
if n_samples < 2 or n_components >= n_features:
    # 样本太少无法拟合 PCA, 或维度已不超过目标 -> 不降维, 直接用原始特征
    print(f"跳过 PCA (样本数={n_samples}, 特征维={n_features}, 目标维={target_dim}), 输出原始维度。")
    reduced_whole = whole_features
else:
    if n_components < target_dim:
        print(f"警告: 样本数量 ({n_samples}) 少于目标维度 {target_dim}, PCA 维度自动调整为 {n_components}")
    print(f"正在进行 PCA 降维 (目标维度: {n_components}) ...")
    reducer = PCA(n_components=n_components)
    reduced_whole = reducer.fit_transform(whole_features).astype(np.float32)
    print(f"PCA 解释方差比 (Total): {np.sum(reducer.explained_variance_ratio_):.4f}")

out_dim = reduced_whole.shape[1]
print(f"whole 降维后特征矩阵形状: {reduced_whole.shape}")

# ----------------------------------------------------------------
# Step 4: 保存 whole 结果
# ----------------------------------------------------------------
np.save(args.fpf_file, reduced_whole)
save_names(args.names_file, whole_names)
print(f"whole 特征已保存: {args.fpf_file} (名称: {args.names_file})")

# ----------------------------------------------------------------
# Step 5 (可选): 用同一个 PCA basis 提取 split contig
# ----------------------------------------------------------------
if args.split_fasta_file is not None:
    print(f"正在读取 split FASTA: {args.split_fasta_file} ...")
    split_names, split_seqs = read_fasta(args.split_fasta_file)
    print(f"共读取到 {len(split_seqs)} 条 split 序列。")
    split_features = encode_sequences(split_seqs)
    if split_features.shape[0] == 0:
        reduced_split = np.zeros((0, out_dim), dtype=np.float32)
    elif reducer is None:
        # whole 未降维, split 也保持原始维度 (两者同在原始 768 维空间)
        reduced_split = split_features.astype(np.float32)
    else:
        # 关键: 用 whole 拟合出的 reducer.transform, 保证 whole 与 split 共享同一 PCA basis
        reduced_split = reducer.transform(split_features).astype(np.float32)
    np.save(args.split_fpf_file, reduced_split)
    save_names(args.split_names_file, split_names)
    print(f"split 特征已保存: {args.split_fpf_file} (名称: {args.split_names_file}); 形状: {reduced_split.shape}")

end_time = time.time()
print("-" * 30)
print(f"处理完成! 耗时: {end_time - start_time:.2f} 秒")
print(f"whole: 名称数={len(whole_names)}, 特征数={len(reduced_whole)}")
if args.split_fasta_file is not None:
    print(f"split: 名称数={len(split_names)}, 特征数={len(reduced_split)}")
    print(f"whole 与 split 已共享同一 PCA basis (维度均为 {out_dim})。")
