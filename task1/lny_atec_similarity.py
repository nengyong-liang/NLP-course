# -*- coding: utf-8 -*-
"""
ATEC 语义相似度任务 - 优化版 + 可视化
方法：TF-IDF vs Word2Vec（仅限这两种方法）
"""

import os
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    precision_recall_curve, roc_curve, auc, confusion_matrix
)
from gensim.models import Word2Vec
import logging
import argparse

# 设置中文字体（防止中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# ========================
# 1. 数据加载与预处理
# ========================

def load_stopwords(stopwords_path=None):
    """加载停用词。优先使用外部文件，否则用内置"""
    if stopwords_path and os.path.exists(stopwords_path):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f if line.strip())
    else:
        # 扩展停用词（可替换为完整停用词表）
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '里', '怎么', '什么', '呢', '吧', '啊', '呀', '吗', '了',
            ' ', '\t', '\n', '，', '。', '；', '：', '？', '！', '“', '”', '（', '）', '【', '】'
        }
    return stopwords

def clean_text(text):
    return text.strip()

def tokenize(text, stopwords):
    words = jieba.lcut(text)
    return [w.strip() for w in words if w.strip() and w not in stopwords and len(w) > 1]

def process_pair(line, stopwords):
    parts = line.strip().split('\t')
    if len(parts) != 3:
        return None
    sent1, sent2, label = parts[0], parts[1], int(parts[2])
    tokens1 = tokenize(clean_text(sent1), stopwords)
    tokens2 = tokenize(clean_text(sent2), stopwords)
    return tokens1, tokens2, label

def load_dataset(file_path, stopwords):
    pairs = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            res = process_pair(line, stopwords)
            if res:
                tokens1, tokens2, label = res
                text1 = ' '.join(tokens1)
                text2 = ' '.join(tokens2)
                pairs.append((text1, text2))
                labels.append(label)
    return pairs, labels

# ========================
# 2. TF-IDF 方法（优化版）
# ========================

def tfidf_similarity_optimized(pairs, threshold=0.5):
    """TF-IDF + 优化参数"""
    all_texts = [text for pair in pairs for text in pair]
    # 优化参数：ngram, sublinear_tf, max_features
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=10000
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarities = []
    n = len(pairs)
    for i in range(n):
        idx1 = i * 2
        idx2 = i * 2 + 1
        sim = cosine_similarity(tfidf_matrix[idx1], tfidf_matrix[idx2])[0][0]
        similarities.append(sim)
    pred_labels = [1 if sim >= threshold else 0 for sim in similarities]
    return pred_labels, similarities

def find_best_threshold(y_true, similarities, method_name="Method"):
    """自动搜索最佳 F1 阈值"""
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        y_pred = [1 if s >= t else 0 for s in similarities]
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    print(f"【{method_name}】最优阈值: {best_thresh:.2f}, 对应 F1: {best_f1:.4f}")
    return best_thresh, best_f1

# ========================
# 3. Word2Vec 方法（优化版）
# ========================

def train_word2vec_optimized(train_file, stopwords, vector_size=200, window=7, min_count=2):
    """优化 Word2Vec 训练参数"""
    sentences = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            res = process_pair(line, stopwords)
            if res:
                tokens1, tokens2, _ = res
                sentences.append(tokens1)
                sentences.append(tokens2)
    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=0  # CBOW
    )
    return model

def sentence_vector(tokens, wv, vector_size):
    vectors = [wv[word] for word in tokens if word in wv]
    if vectors:
        vec = np.mean(vectors, axis=0)
        # 可选：L2 归一化（提升余弦相似度）
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    else:
        return np.zeros(vector_size)

def word2vec_similarity_optimized(pairs, w2v_model, threshold=0.5):
    wv = w2v_model.wv
    vector_size = w2v_model.vector_size
    similarities = []
    for text1, text2 in pairs:
        tokens1 = text1.split()
        tokens2 = text2.split()
        vec1 = sentence_vector(tokens1, wv, vector_size)
        vec2 = sentence_vector(tokens2, wv, vector_size)
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            sim = 0.0
        else:
            sim = np.dot(vec1, vec2)  # 已归一化，即余弦相似度
        similarities.append(sim)
    pred_labels = [1 if sim >= threshold else 0 for sim in similarities]
    return pred_labels, similarities

# ========================
# 4. 可视化函数
# ========================

def plot_similarity_histogram(y_true, sims_tfidf, sims_w2v, save_path="sim_histogram.png"):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist([s for s, y in zip(sims_tfidf, y_true) if y == 1], bins=30, alpha=0.7, label='Positive', color='green')
    plt.hist([s for s, y in zip(sims_tfidf, y_true) if y == 0], bins=30, alpha=0.7, label='Negative', color='red')
    plt.title('TF-IDF 相似度分布')
    plt.xlabel('相似度')
    plt.ylabel('频次')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist([s for s, y in zip(sims_w2v, y_true) if y == 1], bins=30, alpha=0.7, label='Positive', color='green')
    plt.hist([s for s, y in zip(sims_w2v, y_true) if y == 0], bins=30, alpha=0.7, label='Negative', color='red')
    plt.title('Word2Vec 相似度分布')
    plt.xlabel('相似度')
    plt.ylabel('频次')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_pr_roc(y_true, sims_tfidf, sims_w2v, save_path="pr_roc.png"):
    plt.figure(figsize=(12, 5))
    
    # PR Curve
    plt.subplot(1, 2, 1) # 子图
    for name, sims in [("TF-IDF", sims_tfidf), ("Word2Vec", sims_w2v)]:
        precision, recall, _ = precision_recall_curve(y_true, sims) # 计算PR曲线
        pr_auc = auc(recall, precision) # 计算AUC
        plt.plot(recall, precision, label=f'{name} (AUC={pr_auc:.3f})') # 绘制PR曲线
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    # ROC Curve
    plt.subplot(1, 2, 2)
    for name, sims in [("TF-IDF", sims_tfidf), ("Word2Vec", sims_w2v)]:
        fpr, tpr, _ = roc_curve(y_true, sims)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--', alpha=0.5) # 绘制对角线，样式为虚线
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_f1_vs_threshold(y_true, sims_tfidf, sims_w2v, save_path="f1_vs_threshold.png"):
    thresholds = np.arange(0.0, 1.01, 0.01)
    f1_tfidf = []
    f1_w2v = []
    for t in thresholds:
        f1_tfidf.append(f1_score(y_true, [1 if s >= t else 0 for s in sims_tfidf]))
        f1_w2v.append(f1_score(y_true, [1 if s >= t else 0 for s in sims_w2v]))
    
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_tfidf, label='TF-IDF', marker='o', markevery=20) # 样式为
    plt.plot(thresholds, f1_w2v, label='Word2Vec', marker='s', markevery=20)
    plt.xlabel('Threshold')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, method_name, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Similar', 'Similar'], yticklabels=['Not Similar', 'Similar'])
    plt.title(f'混淆矩阵 - {method_name}')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_similarity_scatter(sims_tfidf, sims_w2v, y_true, save_path="sim_scatter.png"):
    plt.figure(figsize=(6,6))
    colors = ['red' if y == 0 else 'green' for y in y_true]
    plt.scatter(sims_tfidf, sims_w2v, c=colors, alpha=0.6, s=10)
    plt.xlabel('TF-IDF 相似度')
    plt.ylabel('Word2Vec 相似度')
    plt.title('两种方法相似度对比（红：负例，绿：正例）')
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.show()

# ========================
# 5. 评估函数
# ========================

def evaluate_and_visualize(y_true, y_pred, sims, method_name, save_prefix=""):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n【{method_name}】性能指标:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    
    # 可视化混淆矩阵
    plot_confusion_matrix(y_true, y_pred, method_name, f"{save_prefix}_cm.png")
    
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}

# ========================
# 6. 主函数
# ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default=r'D:\1_LNY\code\0_data\NLP-course\task1\ATEC\ATEC.train.data')
    parser.add_argument('--test_path',  type=str, default=r'D:\1_LNY\code\0_data\NLP-course\task1\ATEC\ATEC.test.data')
    parser.add_argument('--valid_path', type=str, default=r'D:\1_LNY\code\0_data\NLP-course\task1\ATEC\ATEC.valid.data')
    parser.add_argument('--stopwords_path', type=str, default=None)
    parser.add_argument('--use_valid_for_threshold', action='store_true', help="使用验证集选阈值")
    args = parser.parse_args()

    stopwords = load_stopwords(args.stopwords_path)

    print("加载测试集...")
    test_pairs, test_labels = load_dataset(args.test_path, stopwords)
    print(f"测试集样本数: {len(test_labels)}")

    # 决定用哪个数据集找阈值
    if args.use_valid_for_threshold and os.path.exists(args.valid_path):
        print("使用验证集选择最优阈值...")
        valid_pairs, valid_labels = load_dataset(args.valid_path, stopwords)
        # TF-IDF on valid
        _, valid_sims_tfidf = tfidf_similarity_optimized(valid_pairs)
        best_thresh_tfidf, _ = find_best_threshold(valid_labels, valid_sims_tfidf, "TF-IDF")
        # Word2Vec on valid
        w2v_model = train_word2vec_optimized(args.train_path, stopwords)
        _, valid_sims_w2v = word2vec_similarity_optimized(valid_pairs, w2v_model)
        best_thresh_w2v, _ = find_best_threshold(valid_labels, valid_sims_w2v, "Word2Vec")
    else:
        print("使用测试集自动搜索最优阈值")
        # 先跑一遍获取相似度
        _, test_sims_tfidf = tfidf_similarity_optimized(test_pairs)
        best_thresh_tfidf, _ = find_best_threshold(test_labels, test_sims_tfidf, "TF-IDF")
        w2v_model = train_word2vec_optimized(args.train_path, stopwords)
        _, test_sims_w2v = word2vec_similarity_optimized(test_pairs, w2v_model)
        best_thresh_w2v, _ = find_best_threshold(test_labels, test_sims_w2v, "Word2Vec")

    # === 重新用最优阈值预测 ===
    tfidf_preds, tfidf_sims = tfidf_similarity_optimized(test_pairs, threshold=best_thresh_tfidf)
    w2v_preds, w2v_sims = word2vec_similarity_optimized(test_pairs, w2v_model, threshold=best_thresh_w2v)

    # 评估
    tfidf_metrics = evaluate_and_visualize(test_labels, tfidf_preds, tfidf_sims, "TF-IDF", "tfidf")
    w2v_metrics = evaluate_and_visualize(test_labels, w2v_preds, w2v_sims, "Word2Vec", "w2v")

    # === 结果对比 ===
    print("\n" + "="*60)
    print(f"📊 方法对比 (自动选择最优阈值)")
    print("="*60)
    print(f"{'方法':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Acc':<10}")
    print("-"*60)
    print(f"{'TF-IDF':<12} {tfidf_metrics['precision']:<10.4f} {tfidf_metrics['recall']:<10.4f} {tfidf_metrics['f1']:<10.4f} {tfidf_metrics['accuracy']:<10.4f}")
    print(f"{'Word2Vec':<12} {w2v_metrics['precision']:<10.4f} {w2v_metrics['recall']:<10.4f} {w2v_metrics['f1']:<10.4f} {w2v_metrics['accuracy']:<10.4f}")

    # === 可视化 ===
    plot_similarity_histogram(test_labels, tfidf_sims, w2v_sims, "similarity_histogram.png")
    plot_pr_roc(test_labels, tfidf_sims, w2v_sims, "pr_roc_curves.png")
    plot_f1_vs_threshold(test_labels, tfidf_sims, w2v_sims, "f1_vs_threshold.png")
    plot_similarity_scatter(tfidf_sims, w2v_sims, test_labels, "similarity_scatter.png")

if __name__ == '__main__':
    main()