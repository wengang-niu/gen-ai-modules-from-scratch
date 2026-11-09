"""Flow Matching 数据集定义。"""

import glob
import os

from PIL import Image
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    """CelebA 人脸数据集封装：读取图片 → 预处理 → 缩放到 [-1,1]（与噪声分布对齐）。"""

    def __init__(self, root, transform=None):
        self.root = root
        # KaggleHub 数据集通常包含嵌套目录，因此递归查找所有 JPG 文件。
        self.paths = sorted(glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True))
        if not self.paths:
            raise FileNotFoundError(f"No JPG images found under Kaggle dataset path: {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # 从 [0,1] 缩放到 [-1,1]，与训练时采样的标准正态噪声 x_0 ~ N(0,1) 范围一致
        img = img * 2.0 - 1.0
        return img
