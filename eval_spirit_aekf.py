import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import h5py

# 引用你的模块
from detr.models.spirit_model import SpiritPhysicalCore, VisualFeatureAdapter
from detr.models.backbone import build_backbone
from detr.main import get_args_parser
from utils import get_norm_stats, find_all_hdf5, SpiritPretrainDataset

class SpiritAEKF:
    def __init__(self, state_dim, obs_dim, device):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.device = device
        
        # --- 初始化参数 ---
        # P: 状态协方差矩阵 (不确定性)
        # 初始时我们很不确定，所以给一个较大的对角阵
        self.P = torch.eye(state_dim, device=device) * 0.1
        
        # Q: 过程噪声协方差 (Process Noise)
        # 代表物理预测模型(Net_A)本身的不确定性/误差
        # 如果 Q 很大，P 就会在预测步迅速变大
        self.base_Q = 1e-3
        self.Q = torch.eye(state_dim, device=device) * self.base_Q
        
        # R: 观测噪声协方差 (Measurement Noise)
        # 代表视觉观测(Net_H)的噪声
        # 如果 R 很小，Filter 会非常信任视觉，P 会在更新步迅速变小
        self.R = torch.eye(obs_dim, device=device) * 1e-2
    def set_Q_scale(self, scale):
        # 动态调整 Q：基础值 + 误差带来的放大
        # scale 是从 Loss A 传过来的误差值
        # 这里的系数 100.0 是一个超参数，您可能需要根据 Loss A 的大小（通常是 1e-4 级别）来调
        # 如果 Loss A 是 0.0005，scale*100 = 0.05，Q 就变成了 0.051
        current_Q_val = self.base_Q + scale * 500.0 
        self.Q = torch.eye(self.state_dim, device=self.device) * current_Q_val

    def reset(self):
        self.P = torch.eye(self.state_dim, self.device) * 0.1

    def predict(self, A_mat):
        """
        预测步骤：不确定性增加
        P_pred = A * P * A^T + Q
        """
        # P = A @ P @ A.T + Q
        self.P = torch.bmm(A_mat, torch.bmm(self.P.unsqueeze(0), A_mat.transpose(1, 2))).squeeze(0) + self.Q
        return self.P

    def update(self, H_mat):
        """
        更新步骤：不确定性减少 (前提是能观测到)
        K = P * H^T * (H * P * H^T + R)^-1
        P = (I - K * H) * P
        """
        # 1. 计算卡尔曼增益 K
        # H_P_HT = H @ P @ H.T
        H_P_HT = torch.bmm(H_mat, torch.bmm(self.P.unsqueeze(0), H_mat.transpose(1, 2))).squeeze(0)
        S = H_P_HT + self.R
        
        # S_inv
        try:
            S_inv = torch.linalg.inv(S)
        except RuntimeError:
            # 防止奇异矩阵，加一点点扰动
            S_inv = torch.linalg.inv(S + torch.eye(self.obs_dim, device=self.device) * 1e-6)
            
        # K = P @ H.T @ S_inv
        K = torch.mm(self.P, torch.mm(H_mat.squeeze(0).T, S_inv))
        
        # 2. 更新 P
        # P = (I - K @ H) @ P
        I = torch.eye(self.state_dim, device=self.device)
        KH = torch.mm(K, H_mat.squeeze(0))
        self.P = torch.mm(I - KH, self.P)
        
        return self.P

def calculate_kappa(uncertainties):
    """
    将不确定性 P_t 转换为控制系数 kappa
    """
    P_seq = np.array(uncertainties)
    
    # 1. 确定动态范围
    # 实际应用中这些阈值通常是根据校准阶段确定的
    # 这里我们用数据的统计值来模拟
    P_min = np.percentile(P_seq, 5)   # 基准线 (0.71 左右)
    P_max = np.percentile(P_seq, 95)  # 峰值线 (0.72 左右)
    
    # 防止分母为0
    if P_max == P_min: P_max += 1e-6
    
    # 2. 线性映射: P 越大，Kappa 越小
    # 归一化到 [0, 1]
    norm_P = (P_seq - P_min) / (P_max - P_min)
    
    # 反转并截断
    # alpha 是敏感度系数，alpha 越大，对不确定性越敏感
    alpha = 2.0 
    kappa = 1.0 - np.clip(alpha * norm_P, 0, 0.9) # 保证至少保留 0.1 的动力
    
    return kappa

def evaluate_aekf(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 0. 准备工作
    if not args.camera_names:
        print("⚠️ 警告: 使用默认仿真摄像头 ['top']")
        args.camera_names = ['top']

    # 1. 加载模型
    print(f"Loading models from {args.ckpt_dir}...")
    backbone_model = build_backbone(args).to(device)
    # 冻结 Backbone
    for p in backbone_model.parameters(): p.requires_grad_(False)
        
    visual_adapter = VisualFeatureAdapter(backbone_model, num_cams=len(args.camera_names), feature_dim=64).to(device)
    visual_adapter.load_state_dict(torch.load(os.path.join(args.ckpt_dir, 'visual_adapter.ckpt'), map_location=device))
    visual_adapter.eval()
    
    # 注意 hidden_dim 必须匹配训练时的设置 (256 或 args.hidden_dim)
    # 假设你训练时用的是默认的 256，如果这里报错 shape mismatch，请改成 128 或 args.hidden_dim
    spirit_net = SpiritPhysicalCore(state_dim=14, action_dim=14, obs_feature_dim=64, hidden_dim=args.hidden_dim).to(device)
    spirit_net.load_state_dict(torch.load(os.path.join(args.ckpt_dir, 'spirit_core.ckpt'), map_location=device))
    spirit_net.eval()
    
    # 2. 初始化 AEKF
    ekf = SpiritAEKF(state_dim=14, obs_dim=64, device=device)

    # 3. 准备数据
    dataset_dir_l = [args.dataset_dir]
    dataset_path_list = find_all_hdf5(dataset_dir_l[0], skip_mirrored_data=True)
    norm_stats, _ = get_norm_stats(dataset_path_list)
    
    test_file = dataset_path_list[0]
    print(f"Testing on episode: {test_file}")
    
    # 获取真实长度
    with h5py.File(test_file, 'r') as f:
        real_len = f['/action'].shape[0]
    
    dataset = SpiritPretrainDataset(
        dataset_path_list=[test_file],
        camera_names=args.camera_names,
        norm_stats=norm_stats,
        episode_ids=[0],
        episode_len=[real_len],
        chunk_size=100,
        policy_class='Spirit'
    )
    
    # 4. 运行循环
    metrics = {
        'uncertainty': [],  # trace(P)
        'lambda_c': [],     # 可控性
        'lambda_o': []      # 可观测性
    }
    
    print("Running AEKF loop...")
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            
            # 准备输入
            qpos_t = data['qpos_t'].unsqueeze(0).to(device)
            action_t = data['action_t'][:14].unsqueeze(0).to(device) 
            image_t = data['image_t'].unsqueeze(0).to(device) # [1, num_cams, C, H, W]
            
            # A. 前向传播获取矩阵
            # 视觉特征 o_t
            obs_feat = visual_adapter(image_t) # [1, 64]
            # 雅可比矩阵
            A, B, H = spirit_net.get_jacobians(qpos_t, action_t)
            if i < len(dataset) - 1:
                qpos_next_true = dataset[i+1]['qpos_t'].to(device)
                delta_s_pred = torch.bmm(B, action_t.unsqueeze(-1)).squeeze(-1)
                delta_s_true = qpos_next_true - qpos_t
                dynamics_error = torch.mean((delta_s_pred - delta_s_true)**2).item()
            else:
                dynamics_error = 0.0
            ekf.set_Q_scale(dynamics_error) 
            
            # B. AEKF 步骤
            # 1. Predict
            ekf.predict(A)
            # 2. Update
            P_curr = ekf.update(H)
            
            # C. 记录指标
            # 1. 不确定性 (Trace of P matrix)
            # 也可以用 log determinant: torch.logdet(P_curr)
            uncertainty = torch.trace(P_curr).item()
            metrics['uncertainty'].append(uncertainty)
            
            # 2. 可控性 (lambda_c)
            W_c = torch.bmm(B, B.transpose(1, 2))
            eig_c = torch.linalg.eigvalsh(W_c).cpu().numpy()[0]
            metrics['lambda_c'].append(np.mean(eig_c))
            
            # 3. 可观测性 (lambda_o)
            W_o = torch.bmm(H.transpose(1, 2), H)
            eig_o = torch.linalg.eigvalsh(W_o).cpu().numpy()[0]
            metrics['lambda_o'].append(np.mean(eig_o))
    
    kappa_seq = calculate_kappa(metrics['uncertainty'])
    metrics['kappa'] = kappa_seq
    # 5. 可视化
    plot_metrics(metrics, args.ckpt_dir)



def plot_metrics(metrics, save_dir):
    time_steps = range(len(metrics['uncertainty']))
    
    plt.figure(figsize=(16, 16))
    
    # 1. Uncertainty
    plt.subplot(4, 1, 1)
    plt.plot(time_steps, metrics['uncertainty'], color='purple', linewidth=2)
    plt.title('Introspection Metric 1: Uncertainty (Trace of P_t)')
    plt.ylabel('Uncertainty')
    plt.grid(True)
    # 标出可能的接触区域 (根据你的视频大概是 300-400)
    plt.axvspan(300, 400, color='red', alpha=0.1, label='Contact/Handover Zone')
    plt.legend()
    
    # 2. Controllability
    plt.subplot(4, 1, 2)
    plt.plot(time_steps, metrics['lambda_c'], color='blue')
    plt.title('Introspection Metric 2: Controllability (lambda_c)')
    plt.grid(True)
    
    # 3. Observability
    plt.subplot(4, 1, 3)
    plt.plot(time_steps, metrics['lambda_o'], color='green')
    plt.title('Introspection Metric 3: Observability (lambda_o)')
    plt.xlabel('Time Step')
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.plot(time_steps, metrics['kappa'], color='orange', linewidth=2)
    plt.title('Final Spirit Metric: Control Gain (kappa)')
    plt.ylabel('Scale (0~1)')
    plt.xlabel('Time Step')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True)
    plt.axvspan(300, 400, color='red', alpha=0.1)

    save_path = os.path.join(save_dir, 'full_introspection_aekf.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"图表已保存至: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SPIRIT AEKF', parents=[get_args_parser()])
    parser.add_argument('--dataset_dir', required=True)
    
    args = parser.parse_args()
    if not args.camera_names:
        print("Warning: args.camera_names 为空，使用默认配置 ['left_wrist', 'right_wrist', 'top']")
        args.camera_names = ['left_wrist', 'right_wrist', 'top']
    
    evaluate_aekf(args)