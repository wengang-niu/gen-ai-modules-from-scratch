"""Flow Matching 模型定义。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """时间步 t 的正弦位置编码。

    把标量时间 t 映射为 dim 维向量，正余弦周期性编码能让模型更好地区分不同时间步，
    是扩散模型 / Flow Matching 中注入时间条件的常用做法。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) 浮点数（支持连续时间 t，不要求是整数）
        device = t.device
        half = self.dim // 2
        # 频率按指数递减：freq = 1 / (10000^(i/(half-1)))
        emb = torch.exp(torch.arange(half, device=device) * -(math.log(10000) / (half - 1)))
        # t 与频率外积，得到 (B, half)，再分别取 sin / cos 拼接成 (B, dim)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))  # 若 dim 为奇数，补一个 0 对齐维度
        return emb  # (B, dim)


class ResidualBlock(nn.Module):
    """带时间条件注入的残差块。

    结构：x → conv1 → 加上时间嵌入的投影 → conv2 → 与残差连接相加。
    时间嵌入 t_emb 通过线性层投影到通道数后广播到 (B, C, 1, 1) 加到特征图上，
    从而把“当前时间步”的信息注入每一层。
    """

    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        # 把时间嵌入投影到 out_ch 维，用于在通道上加到特征图上
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        # 输入输出通道不一致时用 1x1 卷积对齐维度，否则恒等
        self.residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        # x: [B, in_ch, H, W]；t_emb: [B, time_emb_dim]
        residual = self.residual_conv(x)  # [B, in_ch, H, W] -> [B, out_ch, H, W]
        h = self.conv1(x)                 # [B, in_ch, H, W] -> [B, out_ch, H, W]
        t_emb = self.time_emb_proj(t_emb) # [B, time_emb_dim] -> [B, out_ch]
        h = h + t_emb[:, :, None, None]   # [B, out_ch, 1, 1] 广播到空间维相加
        h = self.conv2(h)                 # [B, out_ch, H, W] -> [B, out_ch, H, W]
        return h + residual               # [B, out_ch, H, W]


class SelfAttention2D(nn.Module):
    """二维特征图上的多头自注意力。

    把 (B, C, H, W) 特征图 reshape 成 (B, num_heads, C/num_heads, H*W) 的序列，
    在空间位置间做自注意力，用于建模长距离依赖（UNet 底层 / 中间层使用）。
    """

    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, in_channels)
        # 一次 1x1 卷积同时生成 Q、K、V（3 倍通道）
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape           # x: [B, C, H, W]
        h = self.norm(x)               # [B, C, H, W] -> 同尺寸
        qkv = self.qkv(h)              # [B, C, H, W] -> [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # 拆出 Q、K、V，各 [B, C, H, W]
        # reshape 成多头序列：(B, heads, head_dim, H*W)
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        # 缩放点积注意力：softmax(Q^T K / sqrt(d))
        attn = torch.softmax(torch.matmul(q.transpose(-2, -1), k) / math.sqrt(C // self.num_heads), dim=-1)
        # attn: [B, heads, H*W, H*W]
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        # out: [B, heads, head_dim, H*W]
        out = out.contiguous().view(B, C, H, W)  # 还原回 (B, C, H, W)
        out = self.proj_out(out)       # [B, C, H, W] -> [B, C, H, W]
        return x + out  # 残差连接 [B, C, H, W]


class DownBlock(nn.Module):
    """UNet 编码器（下采样）块：若干残差块 + 可选注意力 + 下采样。"""

    def __init__(self, in_ch, out_ch, time_emb_dim, num_blocks=2, downsample=True, use_attention=False):
        super().__init__()
        # 第一个残差块把通道从 in_ch 提升到 out_ch，后续保持 out_ch
        self.blocks = nn.ModuleList([
            ResidualBlock(in_ch if i == 0 else out_ch, out_ch, time_emb_dim)
            for i in range(num_blocks)
        ])
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()
        # stride=2 的卷积实现 2 倍下采样
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1) if downsample else nn.Identity()

    def forward(self, x, t_emb):
        # x: [B, in_ch, H, W]
        skips = []  # 记录各残差块输出，作为上采样阶段的 skip 连接
        for block in self.blocks:
            x = block(x, t_emb)      # 首个块 [B, in_ch, H, W] -> [B, out_ch, H, W]，其余保持
            skips.append(x)          # skip: [B, out_ch, H, W]
        x = self.attn(x)             # [B, out_ch, H, W] -> 同尺寸
        x = self.downsample(x)       # [B, out_ch, H, W] -> [B, out_ch, H/2, W/2]（downsample=False 时不变）
        return x, skips


class UpBlock(nn.Module):
    """UNet 解码器（上采样）块：上采样 + 若干残差块（拼接 skip 连接）+ 可选注意力。"""

    def __init__(self, in_ch, out_ch, time_emb_dim, num_blocks=2, upsample=True, use_attention=False):
        super().__init__()
        # 转置卷积实现 2 倍上采样
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1) if upsample else nn.Identity()
        # 上采样后拼接来自下采样阶段的 skip 特征，所以输入通道 = in_ch(已变 out_ch) + out_ch
        self.blocks = nn.ModuleList([
            ResidualBlock(in_ch + out_ch, out_ch, time_emb_dim)
            for _ in range(num_blocks)
        ])
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()

    def forward(self, x, skips, t_emb):
        # x: [B, in_ch, H, W]（低分辨率，将被上采样）
        # 上采样后空间尺寸记作 H'、W'（正常 upsample 时 H'=2H、W'=2W，upsample=False 时 H'=H、W'=W）
        x = self.upsample(x)         # [B, in_ch, H, W] -> [B, out_ch, H', W']
        for block in self.blocks:
            if skips:
                # 逐个弹出（从最深层开始）的 skip 特征在通道维拼接
                x = torch.cat([x, skips.pop()], dim=1)  # [B, in_ch + out_ch, H', W']
            x = block(x, t_emb)      # [B, in_ch + out_ch, H', W'] -> [B, out_ch, H', W']
        x = self.attn(x)             # [B, out_ch, H', W'] -> 同尺寸
        return x


class MidBlock(nn.Module):
    """UNet 瓶颈（最底层）块：若干残差块 + 自注意力，通道数不变。"""

    def __init__(self, channels, time_emb_dim, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, channels, time_emb_dim)
            for _ in range(num_blocks)
        ])
        self.attn = SelfAttention2D(channels)

    def forward(self, x, t_emb):
        # x: [B, channels, H, W]
        for block in self.blocks:
            x = block(x, t_emb)      # 尺寸不变 [B, channels, H, W] -> 同尺寸
        x = self.attn(x)             # [B, channels, H, W] -> 同尺寸
        return x


class EnhancedUNet(nn.Module):
    """Flow Matching 的主干网络：带时间条件与自注意力的 UNet。

    结构：时间编码 MLP → 初始卷积 → 4 层下采样（通道逐级翻倍）→ 中间瓶颈 → 4 层上采样 → 输出卷积。
    输入：图像 x 与时间 t；输出：预测的速度场（与输入图像同尺寸）。
    """

    def __init__(self, in_ch=3, base_ch=128, time_emb_dim=512, num_res_blocks=2):
        super().__init__()

        # 时间 t → 正弦编码 → MLP → time_emb_dim 维嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch), #  [B, base_ch]
            nn.Linear(base_ch, time_emb_dim), # [B, time_emb_dim]
            nn.SiLU(),#  [B, time_emb_dim]
            nn.Linear(time_emb_dim, time_emb_dim)# [B, time_emb_dim]
        )

        # 初始卷积：通道 in_ch → base_ch，空间尺寸不变（kernel_size=3, padding=1）
        self.init_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # 下采样：通道 base_ch → 2x → 4x → 8x，空间尺寸逐层减半
        self.down1 = DownBlock(base_ch, base_ch, time_emb_dim, num_res_blocks, downsample=False)
        self.down2 = DownBlock(base_ch, base_ch * 2, time_emb_dim, num_res_blocks)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4, time_emb_dim, num_res_blocks)
        self.down4 = DownBlock(base_ch * 4, base_ch * 8, time_emb_dim, num_res_blocks, use_attention=True)

        self.mid = MidBlock(base_ch * 8, time_emb_dim, num_res_blocks * 2)

        # 上采样：通道 8x → 4x → 2x → base_ch，空间尺寸逐层恢复
        self.up4 = UpBlock(base_ch * 8, base_ch * 4, time_emb_dim, num_res_blocks, use_attention=True)
        self.up3 = UpBlock(base_ch * 4, base_ch * 2, time_emb_dim, num_res_blocks)
        self.up2 = UpBlock(base_ch * 2, base_ch, time_emb_dim, num_res_blocks)
        self.up1 = UpBlock(base_ch, base_ch, time_emb_dim, num_res_blocks, upsample=False)

        # 输出层：归一化 + 激活 + 卷积到 3 通道（RGB 速度场）
        self.final = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        # 下面的尺寸注释以默认参数 in_ch=3、base_ch=128、输入 1024x1024 为例。
        # 若实际分辨率不是 1024，只需按同样规律缩放空间尺寸即可（例如 H/2 处为 512）。

        # t: [B] 浮点数，值域 [0,1]
        t_emb = self.time_mlp(t)       # [B] -> [B, 512]
        x = self.init_conv(x)          # [B, 3, 1024, 1024] -> [B, 128, 1024, 1024]

        skips = []  # 收集各下采样层的 skip 特征，供上采样层拼接

        # down1：不改变空间尺寸，两个残差块都输出 [B, 128, 1024, 1024]；
        #       返回 x=[B, 128, 1024, 1024]，并保存 2 个同尺寸 skip。
        x, s1 = self.down1(x, t_emb); skips.extend(s1)

        # down2：残差块输出/skip 为 [B, 256, 1024, 1024]，stride=2 下采样后：
        #       [B, 128, 1024, 1024] -> [B, 256, 512, 512]。
        x, s2 = self.down2(x, t_emb); skips.extend(s2)

        # down3：残差块输出/skip 为 [B, 512, 512, 512]，下采样后：
        #       [B, 256, 512, 512] -> [B, 512, 256, 256]。
        x, s3 = self.down3(x, t_emb); skips.extend(s3)

        # down4：残差块输出/skip 为 [B, 1024, 256, 256]，下采样后：
        #       [B, 512, 256, 256] -> [B, 1024, 128, 128]。
        x, s4 = self.down4(x, t_emb); skips.extend(s4)

        # 瓶颈：空间尺寸和通道数不变。
        x = self.mid(x, t_emb)         # [B, 1024, 128, 128] -> 同尺寸

        # up4：先上采样 [B, 1024, 128, 128] -> [B, 512, 256, 256]；
        #       每次拼接一个 [B, 1024, 256, 256] 的 skip，通道变为 1536，
        #       再经残差块恢复为 [B, 512, 256, 256]（共拼接 2 次）。
        x = self.up4(x, skips, t_emb)

        # up3：先 [B, 512, 256, 256] -> [B, 256, 512, 512]；
        #       与每个 [B, 512, 512, 512] 的 skip 拼接为 [B, 768, 512, 512]，
        #       再恢复为 [B, 256, 512, 512]（共拼接 2 次）。
        x = self.up3(x, skips, t_emb)

        # up2：先 [B, 256, 512, 512] -> [B, 128, 1024, 1024]；
        #       与每个 [B, 256, 1024, 1024] 的 skip 拼接为 [B, 384, 1024, 1024]，
        #       再恢复为 [B, 128, 1024, 1024]（共拼接 2 次）。
        x = self.up2(x, skips, t_emb)

        # up1：不再改变空间尺寸 [B, 128, 1024, 1024]；
        #       与每个 [B, 128, 1024, 1024] 的 skip 拼接为 [B, 256, 1024, 1024]，
        #       再恢复为 [B, 128, 1024, 1024]（共拼接 2 次）。
        x = self.up1(x, skips, t_emb)

        # 输出：归一化/激活不改变尺寸，最后卷积将通道 128 -> 3。
        return self.final(x)            # [B, 128, 1024, 1024] -> [B, 3, 1024, 1024]
