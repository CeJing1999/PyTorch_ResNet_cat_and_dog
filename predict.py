import torch
import os
import numpy as np

from torch import nn
from torchvision import models
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from models.ResNet import ResNetBase
from utils.utils import imageToRGB, resize_image


class MyResNet:
    def __init__(self):
        self.classes = ['cat', 'dog']  # 类别

        '''================
            创建模型
        ================'''
        # self.ResNet = models.resnet18()  # torchvision实现的ResNet18
        # self.ResNet = models.resnet34()  # torchvision实现的ResNet34
        # self.ResNet = models.resnet50()  # torchvision实现的ResNet50
        # fc_input_dim = self.ResNet.fc.in_features
        # self.ResNet.fc = nn.Sequential(
        #     nn.Linear(in_features=fc_input_dim, out_features=2),
        #     nn.Softmax()
        # )

        block = [2, 2, 2, 2]
        channel = [64, 128, 256, 512]
        self.model = ResNetBase(n_blocks=block, n_channels=channel, num_classes=2)  # 自定义实现的ResNet18
        # block = [3, 4, 6, 3]
        # channel = [64, 128, 256, 512]
        # self.model = ResNetBase(n_blocks=block, n_channels=channel, num_classes=2)  # 自定义实现的ResNet34
        # block = [3, 4, 6, 3]
        # channel = [256, 512, 1024, 2048]
        # bottleneck = [64, 128, 256, 512]
        # 自定义实现的ResNet50
        # self.model = ResNetBase(n_blocks=block, n_channels=channel, n_bottlenecks=bottleneck, num_classes=2)

        '''====================
            加载模型权重
        ===================='''
        # state = torch.load('weights/resnet18/best.pth')  # 加载torchvision实现的ResNet18的权重
        # state = torch.load('weights/resnet34/best.pth')  # 加载torchvision实现的ResNet34的权重
        # state = torch.load('weights/resnet50/best.pth')  # 加载torchvision实现的ResNet50的权重
        # self.ResNet.load_state_dict(state['model_state'])

        state = torch.load('weights/ResNet/resnet18/best.pth')  # 加载自定义实现的ResNet18的权重
        # state = torch.load('weights/ResNet/resnet34/best.pth')  # 加载自定义实现的ResNet34的权重
        # state = torch.load('weights/ResNet/resnet50/best.pth')  # 加载自定义实现的ResNet50的权重
        self.model.load_state_dict(state['model_state'])

        # self.ResNet.to('cuda:0')
        # self.ResNet = self.ResNet.eval()  # 使Dropout和BatchNorm无效
        self.model.to('cuda:0')
        self.model = self.model.eval()  # 使Dropout和BatchNorm无效

    def predict_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])  # 计算输入图片的高和宽
        image = imageToRGB(image)
        image_data = resize_image(image, (224, 224), True)
        # 添加上batch_size维度
        image_data = np.expand_dims(np.transpose(np.array(image_data, dtype='float32'), (2, 0, 1)), 0)
        images = torch.from_numpy(image_data)
        images = images.to('cuda:0')
        with torch.no_grad():
            # outputs = self.ResNet(images).max(1)  # 将图像输入torchvision实现的ResNet网络中进行预测
            outputs = self.model(images).max(1)  # 将图像输入自定义实现的ResNet网络中进行预测
            print(outputs)
        predict_class = self.classes[outputs.indices.item()]  # 预测类别
        font = ImageFont.truetype(font='font/font.ttf', size=30)
        draw = ImageDraw.Draw(image)
        draw.text((image_shape[1] / 2, image_shape[0] - 50), text=predict_class, fill='#FF0000', font=font)

        return image, predict_class


if __name__ == '__main__':
    myResNet = MyResNet()
    mode = input('请选择预测模式（pic or dir）：')
    while mode.lower() != 'pic' and mode.lower() != 'dir':
        mode = input('输入无效！请重新输入（pic or dir）：')
    if mode.lower() == 'pic':
        while True:
            filename = input('请输入预测图片文件名：')
            try:
                img = Image.open(filename)
            except Exception as e:
                print(e)
                print('无法打开文件！可能不存在此文件！请检查输入是否正确并重新输入！')
                continue
            else:
                p_image, classname = myResNet.predict_image(img)
                print(filename.rsplit('/')[-1], '===>', classname)
                p_image.show()
                flag = input('\n是否继续？（y/n）')
                if flag.lower() == 'n':
                    break
    elif mode.lower() == 'dir':
        # dir_save_path = 'out/resnet18/'  # 存放torchvision实现的ResNet18的预测结果
        # dir_save_path = 'out/resnet34/'  # 存放torchvision实现的ResNet34的预测结果
        # dir_save_path = 'out/resnet50/'  # 存放torchvision实现的ResNet50的预测结果
        dir_save_path = 'out/ResNet/resnet18/'  # 存放自定义实现的ResNet18的预测结果
        # dir_save_path = 'out/ResNet/resnet34/'  # 存放自定义实现的ResNet34的预测结果
        # dir_save_path = 'out/ResNet/resnet50/'  # 存放自定义实现的ResNet50的预测结果
        dirname = input('请输入预测图片所在目录：')
        img_names = os.listdir(dirname)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(dirname, img_name)
                img = Image.open(image_path)
                r_image, classname = myResNet.predict_image(img)
                print(img_name, classname)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
