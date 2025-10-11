# task_image/classify_image.py
import torch
from torchvision import models, transforms
from PIL import Image

from PIL import Image, ImageDraw
import os

# 自动创建文件夹
os.makedirs("E:/pythonProject1/task_image/task_image", exist_ok=True)

# 创建一张空白图像（RGB 模式，256x256 像素）
img = Image.new("RGB", (256, 256), color=(200, 200, 255))

# 在图像上绘制一个简单的图案
draw = ImageDraw.Draw(img)
draw.rectangle([(50, 50), (200, 200)], outline="red", width=5)
draw.text((80, 100), "TEST", fill="black")

# 保存图像
save_path = "E:/pythonProject1/task_image/task_image/sample.jpg"
img.save(save_path)

print(f"✅ 已生成测试图片：{save_path}")


def main():
    model = models.resnet18(pretrained=True)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    img = Image.open("E:/pythonProject1/task_image/task_image/sample.jpg").convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
    probs = torch.nn.functional.softmax(outputs[0], dim=0)
    top5_prob, top5_catid = torch.topk(probs, 5)

    print("Top5 索引：", top5_catid.tolist())
    print("Top5 概率：", top5_prob.tolist())
    print("说明：上面给出的是 ImageNet 的类别索引。若需要显示类别名称，请把 ImageNet 标签文件（imagenet_classes.txt）放到项目中并映射索引到名称。")

if __name__ == "__main__":
    main()