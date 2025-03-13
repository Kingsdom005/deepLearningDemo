import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from resnet import ResNet18  # 导入 ResNet18 网络
import glob

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
def load_model(model_path, num_classes=10):
    """
    加载预训练的模型
    :param model_path: 模型路径
    :param num_classes: 类别数量
    :return: 加载的模型
    """
    model = ResNet18().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式
    return model

# 图像预处理
def preprocess_image(image_path, resize=64):
    """
    对图像进行预处理，使其符合模型输入要求
    :param image_path: 图像路径
    :param resize: 图像调整大小
    :return: 预处理后的图像张量
    """
    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加 batch 维度
    return image.to(device)

# 预测单张图像
def predict_single_image(model, image_tensor, class_names):
    """
    对单张图像进行预测
    :param model: 加载的模型
    :param image_tensor: 预处理后的图像张量
    :param class_names: 类别名称列表
    :return: 预测的类别名称和概率
    """
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        predicted_class = predicted_class.item()
        confidence = confidence.item()
    return class_names[predicted_class], confidence

# 预测多张图像
def predict_multiple_images(model, image_paths, class_names):
    """
    对多张图像进行预测
    :param model: 加载的模型
    :param image_paths: 图像路径列表
    :param class_names: 类别名称列表
    :return: 每张图像的预测结果
    """
    results = []
    for image_path in image_paths:
        image_tensor = preprocess_image(image_path)
        class_name, confidence = predict_single_image(model, image_tensor, class_names)
        results.append((image_path, class_name, confidence))
    return results

# 主函数
def main():
    # 模型路径
    model_path = "./model/best_model.pth"
    
    # 类别名称（根据你的数据集修改）
    class_names = ["bulbasaur", "charmander", "squirtle", "pikachu", "jigglypuff", 
                   "meowth", "psyduck", "snorlax", "mewtwo", "ditto"]  # 示例类别名称

    # 加载模型
    model = load_model(model_path)

    # 支持用户上传图片
    print("请输入图片路径（支持单张或多张图片，用空格分隔）：")
    image_paths = input().split()

    # 检查图片路径是否存在
    valid_image_paths = []
    for path in image_paths:
        if os.path.exists(path):
            valid_image_paths.append(path)
        else:
            print(f"警告：文件 {path} 不存在，已跳过。")

    if not valid_image_paths:
        print("没有有效的图片路径，程序退出。")
        return

    # 预测图片
    results = predict_multiple_images(model, valid_image_paths, class_names)

    # 输出预测结果
    print("\n预测结果：")
    for image_path, class_name, confidence in results:
        print(f"图片: {image_path} -> 预测类别: {class_name}, 置信度: {confidence:.4f}")

if __name__ == "__main__":
    main()