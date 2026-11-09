"""Flow Matching（整流流 / Rectified Flow）训练脚本兼容入口。"""

from data import CelebADataset
from model import EnhancedUNet
from sampling import sample_flow, save_samples
from train import main

__all__ = [
    "CelebADataset",
    "EnhancedUNet",
    "sample_flow",
    "save_samples",
    "main",
]


if __name__ == "__main__":
    main()
