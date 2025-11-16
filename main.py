from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import Transformer
from utils import load_imdb_data, text_pipeline, preprocess_data

# 加载数据
dataset, vocab = load_imdb_data()

# 预处理训练和测试数据
train_texts, train_labels = preprocess_data(dataset['train'], vocab)
test_texts, test_labels = preprocess_data(dataset['test'], vocab)

# 创建数据加载器
BATCH_SIZE = 32
train_dataset = TensorDataset(train_texts, train_labels)
test_dataset = TensorDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 超参数
EMBEDDING_DIM = 256
LR = 0.001
EPOCH = 3

# 实例化模型
ts = Transformer(
    vocab_size=len(vocab), 
    embed_dim=EMBEDDING_DIM, 
    num_classes=2
).to(device)

print(ts)
print(f"Vocabulary size: {len(vocab)}")

# 定义优化器和损失函数
optimizer = torch.optim.Adam(ts.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# 用于存储训练历史的列表
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# 训练和测试过程
for epoch in range(EPOCH):
    epoch_train_loss = 0.0
    epoch_train_correct = 0
    epoch_total_train = 0
    
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y = b_x.to(device), b_y.to(device)
        
        output = ts(b_x)
        loss = loss_func(output, b_y)
        
        pred_train = torch.argmax(output, dim=1)

        current_correct = (pred_train == b_y).sum().item()
        current_total = b_y.size(0)
        current_loss = loss.item()

        epoch_train_correct += current_correct
        epoch_total_train += current_total
        epoch_train_loss += current_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:  # 每100个批次测试一次
            # 测试集评估
            ts.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0
            
            with torch.no_grad():
                for test_batch_x, test_batch_y in test_loader:
                    test_batch_x, test_batch_y = test_batch_x.to(device), test_batch_y.to(device)
                    test_output = ts(test_batch_x)
                    batch_loss = loss_func(test_output, test_batch_y)
                    test_loss += batch_loss.item()
                    
                    pred_test = torch.argmax(test_output, dim=1)
                    test_correct += (pred_test == test_batch_y).sum().item()
                    test_total += test_batch_y.size(0)
            
            avg_test_loss = test_loss / len(test_loader)
            test_accuracy = test_correct / test_total
            
            # 存储当前指标
            train_losses.append(current_loss)
            train_accuracies.append(current_correct / current_total)
            test_losses.append(avg_test_loss)
            test_accuracies.append(test_accuracy)
            
            ts.train()
            
            current_train_accuracy = current_correct / current_total
            current_train_loss = current_loss
            
            print(f'Epoch: {epoch}, Batch: {step}')
            print(f'Current batch training loss: {current_train_loss:.4f}, Current batch training accuracy: {current_train_accuracy:.4f}')
            print(f'Test loss: {avg_test_loss:.4f}, Test accuracy: {test_accuracy:.4f}\n')
    
    epoch_avg_loss = epoch_train_loss / len(train_loader)
    epoch_accuracy = epoch_train_correct / epoch_total_train
    print(f'Epoch {epoch} completed - Average training loss: {epoch_avg_loss:.4f}, Average training accuracy: {epoch_accuracy:.4f}')

# 绘制训练过程图表
plt.figure(figsize=(15, 5))

# 损失图表
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', alpha=0.7)
plt.plot(test_losses, label='Test Loss', alpha=0.7)
plt.title('Training and Test Loss')
plt.xlabel('Evaluation Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 准确率图表
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy', alpha=0.7)
plt.plot(test_accuracies, label='Test Accuracy', alpha=0.7)
plt.title('Training and Test Accuracy')
plt.xlabel('Evaluation Steps')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# 最终测试
ts.eval()
final_correct = 0
final_total = 0

with torch.no_grad():
    for test_batch_x, test_batch_y in test_loader:
        test_batch_x, test_batch_y = test_batch_x.to(device), test_batch_y.to(device)
        test_output = ts(test_batch_x)
        pred_test = torch.argmax(test_output, dim=1)
        final_correct += (pred_test == test_batch_y).sum().item()
        final_total += test_batch_y.size(0)

final_accuracy = final_correct / final_total
print(f'Final test accuracy: {final_accuracy:.4f}')

# 绘制最终准确率对比图
plt.figure(figsize=(8, 6))
categories = ['Final Test Accuracy']
values = [final_accuracy]
colors = ['lightblue']

bars = plt.bar(categories, values, color=colors, alpha=0.7, edgecolor='blue', linewidth=2)
plt.ylim(0, 1.0)
plt.title('Model Final Test Accuracy', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12)

# 在柱状图上添加数值标签
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('final_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印几个测试样本的预测结果
print("\nTest sample predictions:")
ts.eval()
with torch.no_grad():
    # 获取一些测试样本
    sample_indices = [0, 1, 2, 3, 4]
    sample_texts = []
    sample_labels = []
    
    for idx in sample_indices:
        sample = dataset['test'][idx]
        sample_texts.append(sample['text'])
        sample_labels.append(sample['label'])
    
    sample_tensors = torch.tensor([text_pipeline(text, vocab) for text in sample_texts]).to(device)
    sample_outputs = ts(sample_tensors)
    sample_preds = torch.argmax(sample_outputs, dim=1).cpu().numpy()
    
    for i, idx in enumerate(sample_indices):
        true_label = "Positive" if sample_labels[i] == 1 else "Negative"
        pred_label = "Positive" if sample_preds[i] == 1 else "Negative"
        print(f"Text: {sample_texts[i][:100]}...")

        print(f"True: {true_label}, Predicted: {pred_label}\n")
