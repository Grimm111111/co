import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SkeletonDataset(Dataset):
    def __init__(self, joint_data, bone_data, motion_data, labels, augment=False):
        # 标准化
        self.joint_data = (joint_data - joint_data.mean()) / joint_data.std()
        self.bone_data = (bone_data - bone_data.mean()) / bone_data.std()
        self.motion_data = (motion_data - motion_data.mean()) / motion_data.std()
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        joint = self.joint_data[idx]
        bone = self.bone_data[idx]
        motion = self.motion_data[idx]
        label = self.labels[idx]

        # 如果启用了数据增强，则对 joint 和 bone 数据进行增强
        if self.augment:
            joint, bone = self.augment_data(joint, bone)

        return (
            torch.tensor(joint, dtype=torch.float32),
            torch.tensor(bone, dtype=torch.float32),
            torch.tensor(motion, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

    def augment_data(self, joint, bone):
        """数据增强：对 joint 和 bone 模态数据进行增强"""
        # 添加噪声
        noise_factor = 0.01
        joint += np.random.normal(0, noise_factor, joint.shape)
        bone += np.random.normal(0, noise_factor, bone.shape)

        # 时间序列抖动
        time_jitter = np.random.randint(-5, 5)  # -5 到 5 的随机抖动
        joint = np.roll(joint, time_jitter, axis=1)
        bone = np.roll(bone, time_jitter, axis=1)

        # 关节点随机偏移
        joint_offset_factor = 0.02
        bone_offset_factor = 0.02
        joint += np.random.uniform(-joint_offset_factor, joint_offset_factor, joint.shape)
        bone += np.random.uniform(-bone_offset_factor, bone_offset_factor, bone.shape)

        return joint, bone

def load_data(joint_path, bone_path, motion_path, label_path, batch_size=16, shuffle=True, augment=False):
    # 加载数据
    joint_data = np.load(joint_path)
    bone_data = np.load(bone_path)
    motion_data = np.load(motion_path)
    labels = np.load(label_path)

    # 创建数据集和数据加载器
    dataset = SkeletonDataset(joint_data, bone_data, motion_data, labels, augment=augment)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader
