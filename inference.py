import torch
from datasets import load_dataset                        # 从 HuggingFace datasets下载 MNIST 数据
from transformers import (AutoImageProcessor,            # 用于进行图片的统一格式、归一化
                          ResNetForImageClassification)  # ResNet模型本体


MODEL_NAME = "microsoft/resnet-18"
BATCH_SIZE = 64                      # 每批处理 64 张图
LIMIT = None                         # 先只取前 1000 张图验证流程；跑通后改成 None 就是全部 1 万张

# 选计算设备 mac有相应的mps 如果没有就用cpu即可
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ds 里每个元素长这样：{"image": 一张 28x28 的灰度图, "label": 这张图的正确答案（0~9）}
ds = load_dataset("ylecun/mnist", split="test")
if LIMIT:
    ds = ds.select(range(LIMIT))

# processor：把图片整理成模型要的格式（缩放到 ResNet默认的224、像素值归一化、转成张量）
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# model：下载 ResNet-18 的预训练权重。
# 它原本的最后一层输出 1000 个分数，对应 ImageNet 的 1000 类物体（猫、狗、汽车……）
model = ResNetForImageClassification.from_pretrained(MODEL_NAME)

# 本题关键一步：即把最后一层（分类头）的输出换成"输出 10 个分数"的新层，对应数字 0~9。
# classifier[-1] 是模型里最后那个线性层；in_features 是它原来的输入维度（resnet-18 是 512）。
# 新层是随机初始化的，这就是准确率只有 ~10% 的原因——相当于随便分到十个中的一个。
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 10)

#回到推理的device上
model.to(device).eval()


correct = 0

# no_grad：告诉 PyTorch "这次不算梯度"（只在训练时才需要算），省内存、提速
with torch.no_grad():
    # 从 0 开始，每次跳 BATCH_SIZE 步，切片取一批图。比如 0~63、64~127……
    for i in range(0, len(ds), BATCH_SIZE):
        batch = ds[i : i + BATCH_SIZE]      # 取出这一批（是个字典：image 列表 + label 列表）

        # 把每张图从 28x28 灰度 改造成 224x224 的 RGB 三通道：
        # convert("RGB") → 灰度复制成 3 个通道；resize → 放大到模型要求的尺寸
        images = [im.convert("RGB").resize((224, 224)) for im in batch["image"]]

        # processor 把 PIL 图片列表变成模型能吃的张量，形状 [64, 3, 224, 224]；
        # to(device) 把它送到和模型同一台设备上（否则会报错）
        inputs = processor(images=images, return_tensors="pt").to(device)

        # 前向计算：把图喂给模型，得到 10 个分数（logits）。
        # **inputs 是把字典拆成关键字参数，等价于 model(pixel_values=...)
        # 形状 [64, 10]：64 张图，每张 10 个分数
        preds = model(**inputs).logits.argmax(dim=-1).cpu().tolist()
        #                            ↑ argmax(dim=-1)：在每行的 10 个分数里挑最大的那个的"下标"
       #                              下标正好就是数字本身（0~9）→ 这就是模型的预测答案
        #                              .cpu().tolist()：从 GPU 搬回 CPU，再变成 Python 列表

        # 把这一批的预测和正确答案逐个比，True 当 1 加，统计对了多少张
        correct += sum(p == l for p, l in zip(preds, batch["label"]))

#准确率计算
print(f"images: {len(ds)}, accuracy: {correct / len(ds):.4f}")