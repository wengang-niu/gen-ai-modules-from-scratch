"""Flow Matching 训练入口。"""

import os

import kagglehub
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data import CelebADataset
from model import EnhancedUNet
from sampling import sample_flow, save_samples


def main():
    # ---------------- 用户配置 ----------------
    # 通过 KaggleHub 下载 CelebA 数据集，返回值为数据集缓存目录。
    data_root = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    print("Path to dataset files:", data_root)

    save_dir = "./flowmatch_checkpoints"                 # 模型权重 / 采样图的保存目录
    os.makedirs(save_dir, exist_ok=True)
    batch_size = 16         # 每批样本数
    lr = 1e-4                # 学习率
    num_epochs = 100         # 训练轮数
    image_size = 104         # 图像缩放到 104x104
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有 GPU 用 GPU
    num_workers = 4          # 数据加载的并行进程数
    pin_memory = True        # 锁页内存，加速数据搬运到 GPU
    sample_every = 1         # 每隔多少个 epoch 采样一次生成图像
    num_sample_images = 9    # 每次采样生成的图像数量
    base_ch = 128            # UNet 基础通道数
    flow_steps = 200         # 采样时的积分步数（步数越多越精确，越慢）

    # ---------------- 数据 ----------------
    transform = transforms.Compose([
        transforms.Resize(image_size),      # 缩放到指定尺寸
        transforms.CenterCrop(image_size),  # 中心裁剪成正方形
        transforms.ToTensor(),              # 转为 Tensor，值域 [0,1]
    ])

    dataset = CelebADataset(root=data_root, transform=transform) #数据集
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=pin_memory) #数据加载

    # ---------------- 准备模型、优化器 ----------------
    model = EnhancedUNet(in_ch=3, base_ch=base_ch, time_emb_dim=512, num_res_blocks=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mse = nn.MSELoss()  # 损失函数：预测速度场与真实速度场的 MSE

    # ---------------- 训练循环 ----------------
    print("Starting training... device:", device)
    total_loss = 0.0   # 累计损失，用于统计平均
    step_count = 0     # 累计步数，用于统计平均
    global_step = 0    # 全局步数（跨 epoch）
    for epoch in range(num_epochs):
        for z in loader:
            z = z.to(device)  # 真实图像 x_1，值域 [-1,1]，形状 (B,3,H,W)
            B = z.shape[0]

            # 采样初始噪声 x_0 ~ N(0,1)
            x_0 = torch.randn_like(z)

            # 采样时间 t ~ Uniform(0,1)
            t = torch.rand(B, device=device, dtype=torch.float32)

            # 构造插值点 x_t = t*z + (1-t)*x_0（Flow Matching 的直线路径）
            t_broadcast = t.view(B, 1, 1, 1)
            x_t = t_broadcast * z + (1.0 - t_broadcast) * x_0

            # 真实速度场 u_target = d(x_t)/dt = z - x_0（对直线路径求导）
            u_target = (z - x_0).detach()

            optimizer.zero_grad()

            pred = model(x_t, t)  # 预测速度场，t 形状 (B,)
            loss = mse(pred, u_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
            optimizer.step()
            total_loss += loss.item()
            step_count += 1
            if global_step % 500 == 0:
                avg_loss = total_loss / step_count
                print(f"Epoch {epoch:03d} Step {global_step:06d} Average Loss: {avg_loss:.6f}")
                total_loss = 0.0
                step_count = 0

            global_step += 1
        # 每个采样周期生成一批图像并保存
        if epoch % sample_every == 0:
            samples = sample_flow(model, n_samples=num_sample_images, steps=flow_steps,
                                  image_size=image_size, device=device)
            save_samples(samples, epoch, save_dir)
        # 每个 epoch 结束保存一次 checkpoint
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "epoch": epoch
        }
        ckpt_path = os.path.join(save_dir, f"flowmatch_ckpt_epoch_{epoch:03d}.pt")
        torch.save(ckpt, ckpt_path)
        print(f"[saved checkpoint] {ckpt_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
