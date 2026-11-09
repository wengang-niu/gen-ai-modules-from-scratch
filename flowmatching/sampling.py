"""Flow Matching 采样与存图工具。"""

import math
import os

import torch
from torchvision import utils


def save_samples(x, epoch, save_dir):
    # x: 值域 [-1,1] 的张量，形状 (N,3,H,W)
    out = (x.clamp(-1, 1) + 1.0) / 2.0  # 缩回 [0,1]
    grid = utils.make_grid(out, nrow=int(math.sqrt(out.shape[0]) + 0.999), padding=2)  # 拼成一张网格图
    filename = os.path.join(save_dir, f"sample_epoch_{epoch:03d}.png")
    utils.save_image(grid, filename)
    print(f"[saved] {filename}")


@torch.no_grad()
def sample_flow(model, n_samples, steps, image_size, device):
    """从纯噪声出发，沿学到的速度场欧拉积分生成图像。

    即求解 ODE  dx/dt = u(x, t)，t 从 0 积到 1，初始 x(0) ~ N(0,1)。
    """
    model.eval()
    # 初始状态：标准正态噪声 x ~ N(0,1)
    x = torch.randn(n_samples, 3, image_size, image_size, device=device)
    dt = 1.0 / steps  # 固定步长
    for i in range(steps):
        # 当前时间 t = i/steps，值域 [0,1)
        t = torch.full((n_samples,), float(i) / steps, device=device, dtype=torch.float32)
        u = model(x, t)  # 预测当前状态的速度场
        x = x + u * dt    # 欧拉一步：x_{t+dt} = x_t + u * dt
    model.train()
    return x.clamp(-1, 1)
