import numpy as np

from PIL import Image


def imageToRGB(image):
    """
        将图像转换成RGB图像，防止灰度图在预测时报错。
        代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB。
        :param image: 输入图片
        :return: 转换后的图片
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size, flag=False):
    """
        对输入图像进行resize
        :param image: 输入图片
        :param size: 缩放后大小
        :param flag: 是否对输入图像进行不失真的resize，默认为False
        :return: resize后的图片
    """
    iw, ih = image.size
    w, h = size
    if flag:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def preprocess_input(image):
    """
        对输入图像进行预处理
        :param image: 输入图片
        :return: 预处理后的图片
    """
    image /= 255.0
    return image
