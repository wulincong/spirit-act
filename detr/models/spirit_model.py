import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class SpiritJacobianConfig:
    state_dim = 14
    action_dim = 14
    obs_feature_dim = 64
    hidden_dim = 128

class DynamicsJacobianNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.backbone = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.head_A = nn.Linear(hidden_dim, self.state_dim * self.state_dim)
        self.head_B = nn.Linear(hidden_dim, self.state_dim * self.action_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        feat = self.backbone(x)

        batch_size = state.shape[0]
        A_flat = self.head_A(feat)
        B_flat = self.head_B(feat)

        A_matrix = A_flat.view(batch_size, self.state_dim, self.state_dim)
        B_matrix = B_flat.view(batch_size, self.state_dim, self.action_dim)

        return A_matrix, B_matrix
    

class ObservationJacobianNet(nn.Module):
    def __init__(self, state_dim, obs_feature_dim, hidden_dim):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_feature_dim

        self.backbone = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.head_H = nn.Linear(hidden_dim, self.obs_dim * self.state_dim)

    def forward(self, state):
        feat = self.backbone(state)
        batch_size = state.shape[0]

        H_flat = self.head_H(feat)
        H_matrix = H_flat.view(batch_size, self.obs_dim, self.state_dim)

        return H_matrix
class SpiritPhysicalCore(nn.Module):
    def __init__(self, state_dim, action_dim, obs_feature_dim, hidden_dim):
        super().__init__()
        self.net_A = DynamicsJacobianNet(state_dim, action_dim, hidden_dim)
        self.net_H = ObservationJacobianNet(state_dim, obs_feature_dim, hidden_dim)

        self.net_f = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )
    
    def get_jacobians(self, state, action):
        A, B = self.net_A(state, action)
        H = self.net_H(state)
        return A, B, H

def consistency_loss(net_A, state, action, next_state):
    A, B = net_A(state, action)
    # 预测下一时刻的状态增量
    # delta_s_pred = A * state + B * action (简化版 LTV)
    # 注意：更严谨的做法是 delta_s = A * (0) + B * (action - nominal_action)，
    # 但在训练初期，我们可以简化为拟合 next_state = A @ state + B @ action + bias
    # 这里我们使用 SPIRIT 论文的 Frobenius 范数约束：
    # 比较 "线性预测" 和 "非线性网络/真实数据" 的差异
    
    # 线性预测：
    # s_pred = (A @ state.unsqueeze(-1)).squeeze(-1) + (B @ action.unsqueeze(-1)).squeeze(-1)
    
    # 实际上，训练 Jacobian 最好的方法是让它逼近真实残差：
    delta_s_target = next_state - state
    delta_s_pred = torch.bmm(B, action.unsqueeze(-1)).squeeze(-1)

    return F.mse_loss(delta_s_pred, delta_s_target)


class VisualFeatureAdapter(nn.Module):
    """
    将 ACT 的 Backbone 输出适配为 SPIRIT 需要的低维观测向量 o_t
    输入: [batch, num_cams, 3, H, W]
    输出: [batch, obs_feature_dim] (e.g., 64)
    """
    def __init__(self, backbone_model, num_cams, feature_dim=64):
        super().__init__()
        self.backbone = backbone_model
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.num_cams = num_cams

        self.backbone_out_channels = 512

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projector = nn.Linear(self.backbone_out_channels * num_cams, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, images):
        # images shape: [batch, num_cams, C, H, W]
        batch_size = images.shape[0]

        # 1. 合并 batch 和 cam 维度以便并行通过 backbone
        # [batch * num_cams, C, H, W]
        flat_images = images.view(-1, *images.shape[2:])
        
        # 2. 通过 Backbone (只取 feature map)
        # ACT 的 Joiner 返回 (features, pos_enc)，我们只需要 features
        # features 是一个 NestedTensor 或 list，通常取最后一层
        # 注意：这里需要根据你实际传入的 backbone 类型微调
        # 如果传入的是 Joiner (build_backbone 的返回值):
        features, _ = self.backbone(flat_images) 
        # features[0] 是最后一层的输出 (NestedTensor), .tensors 取出数据
        feature_map = features[0]
        
        # 3. 全局池化 [batch * num_cams, 512, 1, 1] -> [batch * num_cams, 512]
        pooled = self.pool(feature_map).flatten(1)
        
        # 4. 恢复维度并拼接所有摄像头 [batch, num_cams * 512]
        cam_features = pooled.view(batch_size, -1)
        
        # 5. 投影到 SPIRIT 观测空间 [batch, 64]
        obs_vec = self.projector(cam_features)
        obs_vec = self.norm(obs_vec)
        return obs_vec
    
