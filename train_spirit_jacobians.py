import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
import argparse
import sys
import os
import numpy as np
import time
import datetime

# 确保导入路径正确，根据您的文件结构调整
from detr.models.spirit_model import SpiritPhysicalCore, SpiritJacobianConfig, \
                        consistency_loss, VisualFeatureAdapter
from detr.models.backbone import build_backbone 
from detr.main import get_args_parser
from utils import get_norm_stats, SpiritPretrainDataset, find_all_hdf5

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. 准备数据 ---
    # 建议先准备数据，因为 VisualFeatureAdapter 可能需要根据数据统计调整（虽然这里暂时不需要）
    
    # 路径处理
    dataset_dir_l = [args.dataset_dir]
    # 获取文件列表
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data=False) for dataset_dir in dataset_dir_l]
    dataset_path_list = [item for sublist in dataset_path_list_list for item in sublist]
    
    print(f"正在计算统计量，共 {len(dataset_path_list)} 个剧集...")
    norm_stats, all_episode_len = get_norm_stats(dataset_path_list)
    print(f"统计量计算完成。")
    
    num_episodes = len(dataset_path_list)
    episode_ids = np.arange(num_episodes)

    # 【安全检查】处理 camera_names
    # 如果 args.camera_names 为空（默认值），手动指定一套，防止 VisualFeatureAdapter 报错
    if not args.camera_names:
        print("Warning: args.camera_names 为空，使用默认配置 ['left_wrist', 'right_wrist', 'top']")
        args.camera_names = ['left_wrist', 'right_wrist', 'top']

    # 实例化 SpiritPretrainDataset
    dataset = SpiritPretrainDataset(
        dataset_path_list=dataset_path_list,
        camera_names=args.camera_names, 
        norm_stats=norm_stats,
        episode_ids=episode_ids,
        episode_len=all_episode_len,
        chunk_size=args.chunk_size,
        policy_class='Spirit' 
    )

    # 构建 DataLoader
    batch_sampler = BatchSampler(RandomSampler(dataset), batch_size=args.batch_size, drop_last=True)
    dataloader = DataLoader(
        dataset, 
        batch_sampler=batch_sampler, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    # --- 2. 构建模型 ---    
    backbone_model = build_backbone(args).to(device)

    # 视觉适配器 
    num_cams = len(args.camera_names)
    visual_adapter = VisualFeatureAdapter(backbone_model, num_cams=num_cams, feature_dim=64).to(device)
    
    # SPIRIT 核心
    spirit_net = SpiritPhysicalCore(
        state_dim=14, 
        action_dim=14, 
        obs_feature_dim=64, 
        hidden_dim=args.hidden_dim 
    ).to(device)

    trainable_params = [p for p in spirit_net.parameters() if p.requires_grad] + \
                       [p for p in visual_adapter.parameters() if p.requires_grad]
                       
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
    print(f"开始 SPIRIT 预训练... 共 {args.epochs} 个 Epoch") # 修改为 args.epochs
    global_start_time = time.time()
    for epoch in range(args.epochs): # 修改为 args.epochs
        epoch_start_time = time.time()
        total_loss_A = 0
        total_loss_H = 0
        
        # 增加一个计数器用于计算平均 loss
        num_batches = 0
        
        for batch in dataloader:
            # 数据上正轨
            qpos_t = batch['qpos_t'].to(device)       # s_t
            raw_action = batch['action_t'].to(device)   # a_t
            action_t = raw_action[:, :14]   # a_t [batch, 14]
            qpos_next = batch['qpos_next'].to(device) # s_{t+1}
            image_t = batch['image_t'].to(device)     # img_t
            image_next = batch['image_next'].to(device) # img_{t+1}
            
            # --- 前向传播 ---
            
            # 1. 提取观测特征 o_t 和 o_{t+1}
            obs_t = visual_adapter(image_t)
            with torch.no_grad(): 
                obs_next = visual_adapter(image_next)
            
            # 2. 获取雅可比矩阵
            A_mat, B_mat, H_mat = spirit_net.get_jacobians(qpos_t, action_t)
            
            # --- 计算 Loss ---
            
            # Loss A (动力学)
            loss_A = consistency_loss(spirit_net.net_A, qpos_t, action_t, qpos_next)
            
            # Loss H (观测)
            delta_obs_target = obs_next - obs_t
            delta_state = qpos_next - qpos_t
            
            # 预测的观测变化: delta_obs_pred = H @ delta_state
            delta_obs_pred = torch.bmm(H_mat, delta_state.unsqueeze(-1)).squeeze(-1)
            
            loss_H = F.mse_loss(delta_obs_pred, delta_obs_target)
            
            loss = loss_A + loss_H
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(spirit_net.parameters()) + list(visual_adapter.parameters()), 
                max_norm=1.0 # 限制梯度范数最大为 1
            )
            optimizer.step()
            
            total_loss_A += loss_A.item()
            total_loss_H += loss_H.item()
            num_batches += 1
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_elapsed = epoch_end_time - global_start_time
        completed_epochs = epoch + 1
        avg_epoch_duration = total_elapsed / completed_epochs
        remaining_epochs = args.epochs - completed_epochs
        eta_seconds = avg_epoch_duration * remaining_epochs
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

        avg_loss_A = total_loss_A / num_batches if num_batches > 0 else 0
        avg_loss_H = total_loss_H / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch}/{args.epochs}: "
              f"Loss A={avg_loss_A:.6f}, Loss H={avg_loss_H:.6f} | "
              f"用时: {epoch_duration:.1f}s, 预计剩余: {eta_str}")

    # 保存权重
    # 建议保存到 checkpoint 目录
    ckpt_dir = os.path.join(args.ckpt_dir if hasattr(args, 'ckpt_dir') else '.', 'spirit_checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    torch.save(spirit_net.state_dict(), os.path.join(ckpt_dir, 'spirit_core.ckpt'))
    torch.save(visual_adapter.state_dict(), os.path.join(ckpt_dir, 'visual_adapter.ckpt'))
    print(f"模型已保存至 {ckpt_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Spirit DETR', parents=[get_args_parser()])
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset_dir', required=True)
    parser.add_argument('--num_workers', action='store', type=int, help='num_workers', default=1)
    # 如果原 parser 里没有 ckpt_dir，可以加上
    # parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', default='./ckpts')

    # 注意：如果命令行不传 camera_names，args.camera_names 将是 []，这会在 train 函数中被处理
    args = parser.parse_args()
    train(args)
