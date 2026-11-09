# gen-ai-modules-from-scratch

从零实现生成式 AI 模块的集合，各模块独立运行。

- [MoE](./MoE/) — 混合专家：稀疏 Top-k 路由 + 负载均衡，可视化专家特化
- [flowmatching](./flowmatching/) — Flow Matching：UNet 在 CelebA 上训练与采样生成
- [resnet](./resnet/) — 迁移学习：ResNet18 猫狗二分类
- [smolvlm](./smolvlm/) — SmolVLM2 图 → 文推理（需 GPU）

## 依赖

```bash
pip install torch torchvision matplotlib numpy   # MoE / flowmatching / resnet
pip install kagglehub pillow                     # flowmatching
pip install transformers accelerate              # smolvlm
```

运行方式见各目录内脚本：`python xxx.py`。
