from torchsummary import summary
from torchvision import models

from models.ResNet import ResNetBase

if __name__ == '__main__':
    # model = models.resnet18()
    # block = [2, 2, 2, 2]
    # channel = [64, 128, 256, 512]
    # model = ResNetBase(n_blocks=block, n_channels=channel)

    # model = models.resnet34()
    # block = [3, 4, 6, 3]
    # channel = [64, 128, 256, 512]
    # model = ResNetBase(n_blocks=block, n_channels=channel)

    # model = models.resnet50()
    # block = [3, 4, 6, 3]
    # channel = [256, 512, 1024, 2048]
    # bottleneck = [64, 128, 256, 512]
    # model = ResNetBase(n_blocks=block, n_channels=channel, n_bottlenecks=bottleneck)

    # model = models.resnet101()
    # block = [3, 4, 23, 3]
    # channel = [256, 512, 1024, 2048]
    # bottleneck = [64, 128, 256, 512]
    # model = ResNetBase(n_blocks=block, n_channels=channel, n_bottlenecks=bottleneck)

    # model = models.resnet152()
    block = [3, 8, 36, 3]
    channel = [256, 512, 1024, 2048]
    bottleneck = [64, 128, 256, 512]
    model = ResNetBase(n_blocks=block, n_channels=channel, n_bottlenecks=bottleneck)

    model.to('cuda:0')
    summary(model, (3, 224, 224))
