import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# ALOHA / ACT 关节名称定义
JOINT_NAMES = ["Waist", "Shoulder", "Elbow", "Forearm_Roll", "Wrist_Angle", "Wrist_Rotate", "Gripper"]

def plot_zoom(args):
    # 1. 读取数据
    dataset_path = os.path.join(args.dataset_dir, f"episode_{args.episode_idx}.hdf5")
    if not os.path.exists(dataset_path):
        print(f"Error: 文件不存在 {dataset_path}")
        return

    print(f"正在读取: {dataset_path}")
    with h5py.File(dataset_path, 'r') as root:
        qpos = root['/observations/qpos'][()] # (T, 14)
        action = root['/action'][()]         # (T, 14 or 16)

    # 处理 Action 可能存在的 padding (16维转14维)
    action = action[:, :14]

    # 2. 截取 300-400 区间 (根据你的参数可调)
    start_step = args.start
    end_step = args.end
    
    # 防止越界
    max_len = len(qpos)
    end_step = min(end_step, max_len)
    
    qpos_zoom = qpos[start_step:end_step]
    action_zoom = action[start_step:end_step]
    time_steps = np.arange(start_step, end_step)

    # 3. 绘图设置
    # 我们画两张大图：左臂和右臂
    fig, axs = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # --- 左臂 (维度 0-6) ---
    ax_left = axs[0]
    ax_left.set_title(f"Left Arm: Action (Dashed) vs State (Solid) [Step {start_step}-{end_step}]", fontsize=14)
    for i in range(7):
        # 动作 (指令) 用虚线
        color = plt.cm.tab10(i) # 使用系统默认色板区分关节
        ax_left.plot(time_steps, action_zoom[:, i], linestyle='--', alpha=0.7, color=color)
        # 状态 (实际) 用实线
        ax_left.plot(time_steps, qpos_zoom[:, i], linestyle='-', linewidth=2, label=JOINT_NAMES[i], color=color)
    
    ax_left.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_left.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_left.set_ylabel("Joint Position (rad)")

    # --- 右臂 (维度 7-13) ---
    ax_right = axs[1]
    ax_right.set_title(f"Right Arm: Action (Dashed) vs State (Solid) [Step {start_step}-{end_step}]", fontsize=14)
    for i in range(7):
        idx = i + 7
        color = plt.cm.tab10(i)
        ax_right.plot(time_steps, action_zoom[:, idx], linestyle='--', alpha=0.7, color=color)
        ax_right.plot(time_steps, qpos_zoom[:, idx], linestyle='-', linewidth=2, label=JOINT_NAMES[i], color=color)

    ax_right.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_right.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_right.set_ylabel("Joint Position (rad)")
    ax_right.set_xlabel("Time Step")

    # 4. 标记关键事件区域 (可选)
    # 你之前提到的交接大概在 360-370，我们画个框
    for ax in axs:
        ax.axvspan(360, 375, color='yellow', alpha=0.1, label='Potential Handover')

    plt.tight_layout()
    
    save_path = os.path.join(args.dataset_dir, f"zoom_plot_{start_step}_{end_step}.png")
    plt.savefig(save_path)
    print(f"绘图完成！已保存至: {save_path}")
    print("请查看图中实线和虚线的分离情况（Tracking Error）。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True, help='数据集目录')
    parser.add_argument('--episode_idx', required=True, type=int, help='剧集编号 (例如 14)')
    parser.add_argument('--start', type=int, default=300, help='开始步数')
    parser.add_argument('--end', type=int, default=400, help='结束步数')
    
    args = parser.parse_args()
    plot_zoom(args)