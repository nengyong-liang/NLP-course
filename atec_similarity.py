# D:\1_LNY\code\0_data\NLP-course\task1\ATEC\ATEC.test.data
# -*- coding: utf-8 -*-
"""
ATEC 语义相似度任务
方法对比：TF-IDF vs Word2Vec (平均池化)
"""

import os
import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import logging
import argparse

# 设置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# ========================
# 1. 数据加载与预处理
# ========================

def load_stopwords(stopwords_path=None):

    """加载停用词表。若无指定路径，则使用内置常见停用词"""
    
    if stopwords_path and os.path.exists(stopwords_path): # 指定了路径
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f if line.strip()])
    else:
        # 内置简易中文停用词（实际建议下载哈工大/百度/川大停用词表）
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '里', '怎么', '什么', '呢', '吧', '啊', '呀', '吗', '了',
            ' ','\t','\n'
        }
    return stopwords

def clean_text(text):
    """简单文本清洗：去除多余空格等"""
    return text.strip()

def tokenize(text, stopwords):
    """中文分词 + 去停用词"""
    words = jieba.lcut(text)
    words = [w.strip() for w in words if w.strip() and w not in stopwords]
    return words

def process_pair(line, stopwords):
    """处理一行数据：返回 (tokens1, tokens2, label)"""
    parts = line.strip().split('\t')
    if len(parts) != 3:
        return None
    sent1, sent2, label = parts[0], parts[1], int(parts[2])
    tokens1 = tokenize(clean_text(sent1), stopwords)
    tokens2 = tokenize(clean_text(sent2), stopwords)
    return tokens1, tokens2, label

def load_dataset(file_path, stopwords):
    """加载数据集"""
    pairs = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            res = process_pair(line, stopwords)
            if res:
                tokens1, tokens2, label = res
                # 保留原始字符串用于TF-IDF（空格分隔）
                text1 = ' '.join(tokens1)
                text2 = ' '.join(tokens2)
                pairs.append((text1, text2))
                labels.append(label)
    return pairs, labels

# ========================
# 2. TF-IDF 方法
# ========================

def tfidf_similarity(pairs, threshold=0.5):
    """使用 TF-IDF + 余弦相似度计算相似度"""
    all_texts = [text for pair in pairs for text in pair]  # 所有句子
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    similarities = []
    n = len(pairs)
    for i in range(n):
        idx1 = i * 2
        idx2 = i * 2 + 1
        sim = cosine_similarity(tfidf_matrix[idx1], tfidf_matrix[idx2])[0][0]
        similarities.append(sim)
    
    # 转为二分类（0/1）
    pred_labels = [1 if sim >= threshold else 0 for sim in similarities] # 转为二分类
    return pred_labels, similarities

# ========================
# 3. Word2Vec 方法
# ========================

def train_word2vec_from_data(train_file, stopwords, vector_size=100, window=5, min_count=1):
    """从训练集训练 Word2Vec 模型"""
    sentences = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            res = process_pair(line, stopwords)
            if res:
                tokens1, tokens2, _ = res
                sentences.append(tokens1)
                sentences.append(tokens2)
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return model

def sentence_vector(tokens, word2vec_model):
    """通过平均池化得到句向量"""
    vectors = []
    for word in tokens:
        if word in word2vec_model.wv:
            vectors.append(word2vec_model.wv[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

def word2vec_similarity(pairs, word2vec_model, threshold=0.5):
    """使用 Word2Vec + 余弦相似度"""
    pred_labels = []
    similarities = []
    for text1, text2 in pairs:
        tokens1 = text1.split()
        tokens2 = text2.split()
        vec1 = sentence_vector(tokens1, word2vec_model)
        vec2 = sentence_vector(tokens2, word2vec_model)
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            sim = 0.0
        else:
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        similarities.append(sim)
        pred_labels.append(1 if sim >= threshold else 0)
    return pred_labels, similarities

# ========================
# 4. 评估函数
# ========================

def evaluate(y_true, y_pred, method_name):
    """计算 Precision, Recall, F1, Accuracy"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n【{method_name}】性能指标:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}

# ========================
# 5. 主函数
# ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default=r'D:\1_LNY\code\0_data\NLP-course\task1\ATEC\ATEC.train.data')
    parser.add_argument('--test_path',  type=str, default=r'D:\1_LNY\code\0_data\NLP-course\task1\ATEC\ATEC.test.data')
    parser.add_argument('--valid_path', type=str, default=r'D:\1_LNY\code\0_data\NLP-course\task1\ATEC\ATEC.valid.data')
    parser.add_argument('--stopwords_path', type=str, default=None)  # 可选
    parser.add_argument('--threshold', type=float, default=0.5) #代表阈值，用于确定相似度
    args = parser.parse_args()

    # 加载停用词
    stopwords = load_stopwords(args.stopwords_path)

    # 加载测试集
    print("加载测试集...")
    test_pairs, test_labels = load_dataset(args.test_path, stopwords)
    print(f"测试集样本数: {len(test_labels)}")

    # === 方法1: TF-IDF ===
    print("\n【方法1：TF-IDF + 余弦相似度】")
    tfidf_preds, tfidf_sims = tfidf_similarity(test_pairs, threshold=args.threshold)
    tfidf_metrics = evaluate(test_labels, tfidf_preds, "TF-IDF")

    # === 方法2: Word2Vec (从训练集训练) ===
    print("\n【方法2：Word2Vec (训练) + 平均池化 + 余弦相似度】")
    print("正在训练 Word2Vec 模型（仅用训练集）...")
    w2v_model = train_word2vec_from_data(args.train_path, stopwords)
    w2v_preds, w2v_sims = word2vec_similarity(test_pairs, w2v_model, threshold=args.threshold)
    w2v_metrics = evaluate(test_labels, w2v_preds, "Word2Vec")

    # === 结果对比 ===
    print("\n" + "="*50)
    print("📊 方法对比 (Threshold = {:.2f})".format(args.threshold))
    print("="*50)
    print(f"{'方法':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Acc':<10}")
    print("-"*50)
    print(f"{'TF-IDF':<12} {tfidf_metrics['precision']:<10.4f} {tfidf_metrics['recall']:<10.4f} {tfidf_metrics['f1']:<10.4f} {tfidf_metrics['accuracy']:<10.4f}")
    print(f"{'Word2Vec':<12} {w2v_metrics['precision']:<10.4f} {w2v_metrics['recall']:<10.4f} {w2v_metrics['f1']:<10.4f} {w2v_metrics['accuracy']:<10.4f}")

if __name__ == '__main__':
    main()

