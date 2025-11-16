from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
# 加载IMDB电影评论数据集（情感分析二分类）
def load_imdb_data():
    dataset = load_dataset('imdb')
    
    # 构建词汇表
    word_counts = {}
    for split in ['train', 'test']:
        for example in dataset[split]:
            text = example['text'].lower()
            words = text.split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
    
    # 选择最常见的词汇
    vocab_size = 10000
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(sorted_words[:vocab_size-2])}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    
    return dataset, vocab

# 文本处理管道
def text_pipeline(text, vocab, max_length=512):
    words = text.lower().split()[:max_length]
    indices = [vocab.get(word, vocab['<unk>']) for word in words]
    # 填充到固定长度
    if len(indices) < max_length:
        indices.extend([vocab['<pad>']] * (max_length - len(indices)))
    else:
        indices = indices[:max_length]
    return indices
# 数据预处理
def preprocess_data(dataset_split, vocab):
    texts = []
    labels = []
    
    for example in dataset_split:
        text_indices = text_pipeline(example['text'], vocab)
        texts.append(text_indices)
        labels.append(example['label'])
    
    return torch.tensor(texts, dtype=torch.long), torch.tensor(labels, dtype=torch.long)