from transformers import AutoProcessor 
from transformers import AutoModelForImageTextToText
import torch

model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
processor = AutoProcessor.from_pretrained(model_path)

model = AutoModelForImageTextToText.from_pretrained(
    model_path,                         # 去 HuggingFace 下载 SmolVLM2 这个模型
    torch_dtype=torch.bfloat16,         # 用 bfloat16 半精度存储（省显存、快）
    _attn_implementation="sdpa"         # 注意力用 PyTorch 优化的 SDPA 实现（加速）
).to("cuda")                            # 加载完搬到 GPU 上


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Can you describe this image?"},
        ]
    },
]

inputs = processor.apply_chat_template(   # 套聊天模板 + 转张量
    messages,                             # ① 输入：那段对话
    add_generation_prompt=True,           # ② 加"模型开始回答"的提示
    tokenize=True,                        # ③ 文字转 token id
    return_dict=True,                     # ④ 返回字典
    return_tensors="pt",                  # ⑤ 返回 PyTorch 张量
).to(model.device, dtype=torch.bfloat16)  # ⑥ 搬到 GPU + 转精度
# inputs 是个字典，常用键的维度：
#   input_ids:       [batch=1, seq_len]      → 文字的 token id
#   attention_mask:  [batch=1, seq_len]      → 标记哪些位置是有效文字
#   pixel_values:    [batch=1, 图片数, C=3, H, W]  → 图像的像素张量

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64) #模型生成答案
# generated_ids 形状: [batch=1, seq_len]  =  输入的 token + 新生成的 64 个 token 拼接在一起

generated_texts = processor.batch_decode(
    generated_ids,             # generate 输出的 token id 张量，形状 [batch=1, seq_len]
    skip_special_tokens=True,  # 跳过 <|im_start|>、<image>、<|im_end|> 等特殊标记，只留正常文字
)
# batch_decode 按 batch 维度逐行解码 → 返回字符串 list，长度 = batch=1
# 所以用 [0] 取出唯一那句

print(generated_texts[0])
