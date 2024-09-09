import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(GraphConvolution, self).__init__()
        self.A = nn.Parameter(A, requires_grad=True)  # 邻接矩阵作为可学习参数
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.to(x.device)  # 邻接矩阵移动到相同设备上
        x = torch.einsum('nctv,vw->nctw', (x, A))  # 图卷积
        x = self.conv(x)
        return self.relu(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=-1, keepdim=True)
        max_out = torch.max(x, dim=-1, keepdim=True)[0]
        out = self.fc1(avg_out + max_out)
        out = self.fc2(F.relu(out))
        return torch.sigmoid(out)


class AGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(AGCNBlock, self).__init__()
        self.gcn = GraphConvolution(in_channels, out_channels, A)
        self.att = ChannelAttention(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.gcn(x)
        x = x * self.att(x)  # 使用注意力机制调整通道权重
        return self.relu(x)


class TwoStreamAGCN(nn.Module):
    def __init__(self, num_class, num_point, num_person, in_channels, A, dropout_rate=0.5):
        super(TwoStreamAGCN, self).__init__()
        self.A = A
        self.data_bn = nn.BatchNorm2d(in_channels)

        # Joint Stream
        self.joint_layer1 = AGCNBlock(in_channels, 64, A)
        self.joint_layer2 = AGCNBlock(64, 128, A)
        self.joint_layer3 = AGCNBlock(128, 256, A)

        # Bone Stream
        self.bone_layer1 = AGCNBlock(in_channels, 64, A)
        self.bone_layer2 = AGCNBlock(64, 128, A)
        self.bone_layer3 = AGCNBlock(128, 256, A)

        # Motion Stream
        self.motion_layer1 = AGCNBlock(in_channels, 64, A)
        self.motion_layer2 = AGCNBlock(64, 128, A)
        self.motion_layer3 = AGCNBlock(128, 256, A)

        # Dropout and Fully Connected Layer
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(768, num_class)  # 768 = 256 (joint) + 256 (bone) + 256 (motion)

    def forward(self, joint, bone, motion):
        # 调整 joint, bone, motion 的维度 (N, C, T, V, M) -> (N*M, C, T, V)
        N, C, T, V, M = joint.size()

        joint = joint.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        bone = bone.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        motion = motion.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        # Joint Stream Forward
        joint = self.joint_layer1(joint)
        joint = self.joint_layer2(joint)
        joint = self.joint_layer3(joint)
        joint = F.avg_pool2d(joint, (joint.size(2), joint.size(3)))  # Global Pooling for Joint

        # Bone Stream Forward
        bone = self.bone_layer1(bone)
        bone = self.bone_layer2(bone)
        bone = self.bone_layer3(bone)
        bone = F.avg_pool2d(bone, (bone.size(2), bone.size(3)))  # Global Pooling for Bone

        # Motion Stream Forward
        motion = self.motion_layer1(motion)
        motion = self.motion_layer2(motion)
        motion = self.motion_layer3(motion)
        motion = F.avg_pool2d(motion, (motion.size(2), motion.size(3)))  # Global Pooling for Motion

        # Concatenate Joint, Bone, and Motion Streams
        joint = joint.view(joint.size(0), -1)  # Flatten Joint
        bone = bone.view(bone.size(0), -1)     # Flatten Bone
        motion = motion.view(motion.size(0), -1)  # Flatten Motion

        fused = torch.cat([joint, bone, motion], dim=1)  # Concatenate all features

        # Apply Dropout and Fully Connected Layer
        fused = self.dropout(fused)
        output = self.fc(fused)

        # 恢复 batch_size 维度为 N
        output = output.view(N, M, -1).mean(dim=1)  # (N, M, num_class) -> (N, num_class)

        return output


