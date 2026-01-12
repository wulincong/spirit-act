import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

# 引用你的模块
from detr.models.spirit_model import SpiritPhysicalCore, VisualFeatureAdapter
from detr.models.backbone import build_backbone
from detr.main import get_args_parser
from utils import get_norm_stats, find_all_hdf5, SpiritPretrainDataset

def compute_intrinsic_metrics(A, B, H):
    """
    计算物理内省指标
    """
    # 1. 可控性 (Controllability): 基于 B 矩阵
    # W_c = B * B^T
    # 特征值越大，说明该维度越容易被动作改变
    W_c = torch.bmm(B, B.transpose(1, 2))
    # 计算特征值 (Eigenvalues)，只取实部
    eigen_c = torch.linalg.eigvalsh(W_c) 
    
    # 2. 可观测性 (Observability): 基于 H 矩阵
    # W_o = H^T * H
    # 特征值越大，说明该维度的状态变化越容易引起观测值的变化
    W_o = torch.bmm(H.transpose(1, 2), H)
    eigen_o = torch.linalg.eigvalsh(W_o)
    
    return eigen_c, eigen_o

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 1. 加载模型 ---
    print(f"正在加载模型从: {args.ckpt_dir} ...")
    
    # Backbone
    backbone_model = build_backbone(args).to(device)
    # Visual Adapter
    visual_adapter = VisualFeatureAdapter(backbone_model, num_cams=3, feature_dim=64).to(device)
    adapter_path = os.path.join(args.ckpt_dir, 'visual_adapter.ckpt')
    visual_adapter.load_state_dict(torch.load(adapter_path, map_location=device))
    visual_adapter.eval() # 开启评估模式
    
    # SPIRIT Core
    spirit_net = SpiritPhysicalCore(state_dim=14, action_dim=14, obs_feature_dim=64, hidden_dim=args.hidden_dim).to(device)
    core_path = os.path.join(args.ckpt_dir, 'spirit_core.ckpt')
    spirit_net.load_state_dict(torch.load(core_path, map_location=device))
    spirit_net.eval()

    # --- 2. 加载单条数据进行测试 ---
    # 我们只取验证集的第一条数据来看看
    dataset_dir_l = [args.dataset_dir]
    dataset_path_list = find_all_hdf5(dataset_dir_l[0], skip_mirrored_data=True)
    norm_stats, _ = get_norm_stats(dataset_path_list)
    
    # 手动读取第一个文件
    test_file = dataset_path_list[0]
    print(f"正在分析剧集: {test_file}")
    
    # 临时构建 Dataset 对象来复用读取逻辑
    # 注意：这里只为了读数据，不做 DataLoader
    import h5py
    with h5py.File(test_file, 'r') as f:
        # 读取 action 或 qpos 的长度作为真实长度
        real_len = f['/action'].shape[0]
    
    print(f"检测到真实剧集长度: {real_len}")

    dataset = SpiritPretrainDataset(
        dataset_path_list=[test_file],
        camera_names=args.camera_names,
        norm_stats=norm_stats,
        episode_ids=[0],
        episode_len=[real_len], # dummy
        chunk_size=100,
        policy_class='Spirit'
    )
    
    # 提取整个 episode 的数据
    trajectory_len = len(dataset)
    print(f"剧集长度: {trajectory_len} 帧")
    
    all_eigen_c = []
    all_eigen_o = []
    pred_errors_s = [] # 状态预测误差
    
    with torch.no_grad():
        for i in range(trajectory_len - 1):
            data = dataset[i] # 获取单个时间步
            
            # 准备输入 (增加 batch 维度)
            qpos_t = data['qpos_t'].unsqueeze(0).to(device)
            # 记得切片 action!
            action_t = data['action_t'][:14].unsqueeze(0).to(device) 
            qpos_next_true = data['qpos_next'].unsqueeze(0).to(device)
            
            # --- 前向传播 ---
            A, B, H = spirit_net.get_jacobians(qpos_t, action_t)
            
            # --- 指标计算 ---
            eig_c, eig_o = compute_intrinsic_metrics(A, B, H)
            
            # 保存均值作为代表 (或者你可以保存 max/min)
            # 这里我们保存所有维度的平均可控性/可观测性，便于画图
            all_eigen_c.append(eig_c.cpu().numpy()[0]) # [14]
            all_eigen_o.append(eig_o.cpu().numpy()[0]) # [14]
            
            # --- 验证 Loss A (预测准确性) ---
            # s_next_pred = s_t + B * a_t (简化验证)
            delta_s_pred = torch.bmm(B, action_t.unsqueeze(-1)).squeeze(-1)
            delta_s_true = qpos_next_true - qpos_t
            error = torch.mean((delta_s_pred - delta_s_true)**2).item()
            pred_errors_s.append(error)

    # --- 3. 可视化绘图 ---
    all_eigen_c = np.array(all_eigen_c) # shape [T, 14]
    all_eigen_o = np.array(all_eigen_o) # shape [T, 14]
    
    time_steps = range(len(all_eigen_c))
    
    plt.figure(figsize=(15, 10))
    
    # 子图 1: 状态预测误差 (检查网络是否学到了物理规律)
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, pred_errors_s, label='State Prediction MSE', color='red')
    plt.title('Validation: Dynamics Prediction Error (Loss A check)')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    # 子图 2: 可控性指标 (Controllability)
    plt.subplot(3, 1, 2)
    # 画出所有关节的平均可控性，或者画出特定关节
    plt.plot(time_steps, np.mean(all_eigen_c, axis=1), label='Mean Controllability', color='blue', linewidth=2)
    # 也可以画出最大/最小值的包络
    plt.fill_between(time_steps, np.min(all_eigen_c, axis=1), np.max(all_eigen_c, axis=1), color='blue', alpha=0.1)
    plt.title('Introspection: Physical Controllability (lambda_c)')
    plt.ylabel('Score (Log Scale recommended if spread large)')
    plt.grid(True)
    
    # 子图 3: 可观测性指标 (Observability)
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, np.mean(all_eigen_o, axis=1), label='Mean Observability', color='green', linewidth=2)
    plt.fill_between(time_steps, np.min(all_eigen_o, axis=1), np.max(all_eigen_o, axis=1), color='green', alpha=0.1)
    plt.title('Introspection: Visual Observability (lambda_o)')
    plt.ylabel('Score')
    plt.xlabel('Time Step')
    plt.grid(True)
    
    save_path = os.path.join(args.ckpt_dir, 'introspection_analysis.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n分析完成！结果图已保存至: {save_path}")
    print("请打开图片查看指标变化曲线。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SPIRIT Eval', parents=[get_args_parser()])
    parser.add_argument('--dataset_dir', action='store', type=str, required=True)

    args = parser.parse_args()
    if not args.camera_names:
        print("Warning: args.camera_names 为空，使用默认配置 ['left_wrist', 'right_wrist', 'top']")
        args.camera_names = ['left_wrist', 'right_wrist', 'top']
    evaluate(args)