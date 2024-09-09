import torch
import torch.nn as nn
import numpy as np
import logging
import time
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import TwoStreamAGCN
from data_loader import load_data

# 设置日志文件
logging.basicConfig(filename='train_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# 加载数据
train_loader = load_data(
    'D:\\AI Competition\\基于无人机的人体行为识别-参赛资源(省赛)\\data\\train_joint.npy',
    'D:\\AI Competition\\基于无人机的人体行为识别-参赛资源(省赛)\\data\\train_bone.npy',
    'D:\\AI Competition\\基于无人机的人体行为识别-参赛资源(省赛)\\data\\train_joint_motion.npy',  # 加载 motion 数据
    'D:\\AI Competition\\基于无人机的人体行为识别-参赛资源(省赛)\\data\\train_label.npy',
    batch_size=8
)

test_loader = load_data(
    'D:\\AI Competition\\基于无人机的人体行为识别-参赛资源(省赛)\\data\\test_joint.npy',
    'D:\\AI Competition\\基于无人机的人体行为识别-参赛资源(省赛)\\data\\test_bone.npy',
    'D:\\AI Competition\\基于无人机的人体行为识别-参赛资源(省赛)\\data\\test_joint_motion.npy',  # 加载 motion 数据
    'D:\\AI Competition\\基于无人机的人体行为识别-参赛资源(省赛)\\data\\test_label.npy',
    shuffle=False
)

# 初始化模型
A = torch.eye(17, dtype=torch.float32)  # 邻接矩阵
model = TwoStreamAGCN(num_class=155, num_point=17, num_person=2, in_channels=3, A=A, dropout_rate=0.5).cuda()

# 定义损失函数和优化器（包含 L2 正则化）
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)

# 初始化 TensorBoard
writer = SummaryWriter(log_dir='runs/experiment_name')

# 训练模型
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=100):
    best_accuracy = 0.0  # 初始化最佳准确率

    for epoch in range(epochs):
        start_time = time.time()
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for i, (joint, bone, motion, labels) in enumerate(train_loader):
            joint, bone, motion, labels = joint.cuda(), bone.cuda(), motion.cuda(), labels.cuda()

            # 前向传播
            output = model(joint, bone, motion)
            loss = criterion(output, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算训练集精度
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 每100步记录一次损失
            if i % 100 == 99:
                writer.add_scalar('Training Loss', running_loss / 100, epoch * len(train_loader) + i)
                logging.info(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

        # 记录每个 epoch 的训练集精度
        train_accuracy = 100 * correct / total
        writer.add_scalar('Training Accuracy', train_accuracy, epoch)
        logging.info(f'Epoch [{epoch + 1}/{epochs}], Training Accuracy: {train_accuracy:.2f}%')

        # 测试模型并记录准确率
        test_accuracy, confidences = test_model_with_confidence(model, test_loader)
        writer.add_scalar('Test Accuracy', test_accuracy, epoch)
        logging.info(f'Epoch [{epoch + 1}/{epochs}], Test Accuracy: {test_accuracy:.2f}%')

        # 调整学习率
        scheduler.step(test_accuracy)

        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            np.save('D:\\AI Competition\\基于无人机的人体行为识别-参赛资源(省赛)\\eval\\pred.npy', confidences)
            torch.save(model.state_dict(), 'model_best.pth')
            logging.info(f'保存了新的最佳模型！测试准确率: {best_accuracy:.2f}%')

        epoch_duration = time.time() - start_time
        logging.info(f'Epoch [{epoch + 1}/{epochs}] completed in {epoch_duration:.2f} seconds.')

    writer.close()

# 测试模型并返回置信度和准确率
def test_model_with_confidence(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_confidences = []

    with torch.no_grad():
        for joint, bone, motion, labels in test_loader:
            joint, bone, motion, labels = joint.cuda(), bone.cuda(), motion.cuda(), labels.cuda()

            # 前向传播
            output = model(joint, bone, motion)

            # 计算置信度
            confidence = torch.softmax(output, dim=1)
            all_confidences.append(confidence.cpu().numpy())

            # 计算预测类别
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    all_confidences = np.vstack(all_confidences)

    return accuracy, all_confidences

# 训练和测试
train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=100)

