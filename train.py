import torch
import os
import json

from torch import nn, optim
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from models.ResNet import ResNetBase


def save_model(path, filename, net, optimizer=None, loss=None):
    """
        保存模型
        :param path: 模型保存路径
        :param filename: 模型文件名
        :param net: 网络模型
        :param optimizer: 模型优化器，默认为None，表示不保存优化器
        :param loss: 模型损失，默认为None，表示不保存损失
    """
    state = {
        'model_state': net.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else {},
        'loss': loss if loss is not None else {}
    }
    torch.save(state, os.path.join(path, filename))


def eval_net(net, loader, device='cpu'):
    """
        验证网络
        :param net: 网咯模型
        :param loader: 验证数据DataLoader
        :param device: 计算设备，默认为cpu
        :return: 验证集精度
    """
    net.eval()  # 使Dropout和BatchNorm无效
    y_truth = []
    y_preds = []
    for i, (x, y) in tqdm(enumerate(loader), total=len(loader)):
        # 通过to方法传递到执行计算的设备
        x = x.to(device)
        y = y.to(device)
        # 预测概率最大的类
        with torch.no_grad():
            _, y_pred = net(x).max(1)
        y_truth.append(y)
        y_preds.append(y_pred)
    # 汇总每个小批量预测结果
    y_truth = torch.cat(y_truth)
    y_preds = torch.cat(y_preds)
    acc = (y_truth == y_preds).float().sum() / len(y_truth)  # 计算预测精度

    return acc.item()


def train_net(net, train_loader, test_loader, optimizer_cls=optim.Adam, loss_fn=nn.CrossEntropyLoss(), epochs=100,
              device='cpu', writer=None, weights_path=None):
    """
        训练网络
        :param net: 网络模型
        :param train_loader: 训练数据DataLoader
        :param test_loader: 验证数据DataLoader
        :param optimizer_cls: 模型优化器，默认为Adam
        :param loss_fn: 损失函数，默认为CrossEntropyLoss
        :param epochs: 总的训练轮数，默认为100
        :param device: 计算设备，默认为cpu
        :param writer: 记录日志的SummaryWriter对象，默认为None，表示不记录日志
        :param weights_path: 模型权重保存路径，默认为None，表示不保存模型权重
        :return: [最佳训练轮数，最佳训练损失，最佳训练精度，最佳验证精度]
    """
    best_loss = 1.0  # 最佳损失
    best_acc = 0.0  # 最佳精度
    train_losses = []
    train_acc = []
    val_acc = []
    optimizer = optimizer_cls(net.parameters())
    for epoch in range(epochs):
        train_loss = 0.0
        net.train()  # 将网络设为训练模式
        n = 0
        n_acc = 0
        for i, (xx, yy) in tqdm(enumerate(train_loader), total=len(train_loader)):
            xx = xx.to(device)
            yy = yy.to(device)
            pred = net(xx)
            loss = loss_fn(pred, yy)  # 计算损失
            optimizer.zero_grad()  # 删除上一轮训练的梯度值
            loss.backward()  # 计算损失梯度
            optimizer.step()  # 更新参数
            train_loss += loss.item()
            n += len(xx)
            _, y_pred = pred.max(1)
            n_acc += (yy == y_pred).float().sum().item()
        train_losses.append(train_loss / i)  # 训练数据的损失
        train_acc.append(n_acc / n)  # 训练数据的预测精度
        val_acc.append(eval_net(net, test_loader, device))  # 验证数据的预测精度
        print(epoch, train_losses[-1], train_acc[-1], val_acc[-1], flush=True)  # 显示本轮训练结果
        if writer is not None:
            writer.add_scalar('train_loss', train_losses[-1], epoch)
            writer.add_scalars('accuracy', {'train': train_acc[-1], 'val': val_acc[-1]}, epoch)
        if best_loss > train_losses[-1] and weights_path is not None:
            save_model(weights_path, 'best_loss.pth', net)
            best_loss = train_losses[-1]
        if best_acc < val_acc[-1] and weights_path is not None:
            save_model(weights_path, 'best_acc.pth', net)
            best_acc = val_acc[-1]
        # if best_loss > train_losses[-1] and best_acc < val_acc[-1] and weights_path is not None:
        #     save_model(weights_path, 'best.pth', net)

    return {'best_loss': best_loss, 'best_acc': best_acc}


if __name__ == '__main__':
    '''========================================
        1.通过ImageFolder函数创建Dataset
    ========================================'''
    train_imgs = ImageFolder(
        'datasets/cat_and_dog/train',
        transform=transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=224),
                transforms.ToTensor()
            ]
        )
    )
    print(train_imgs.classes, train_imgs.class_to_idx)
    test_imgs = ImageFolder(
        'datasets/cat_and_dog/test',
        transform=transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=224),
                transforms.ToTensor()
            ]
        )
    )
    print(test_imgs.classes, test_imgs.class_to_idx)

    '''========================
        2.创建DataLoader
    ========================'''
    train_loader = DataLoader(train_imgs, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_imgs, batch_size=16, shuffle=False)
    print('训练集批次数量：', train_loader.__len__(), '；验证集批次数量：', test_loader.__len__())

    '''==================
        3.创建模型
    =================='''
    # ResNet = models.resnet18()  # torchvision实现的ResNet18
    # ResNet = models.resnet34()  # torchvision实现的ResNet34
    # ResNet = models.resnet50()  # torchvision实现的ResNet50
    # fc_input_dim = ResNet.fc.in_features
    # ResNet.fc = nn.Sequential(
    #     nn.Linear(in_features=fc_input_dim, out_features=2),
    #     nn.Softmax(dim=1)
    # )

    # block = [2, 2, 2, 2]
    # channel = [64, 128, 256, 512]
    # model = ResNetBase(n_blocks=block, n_channels=channel, num_classes=2)  # 自定义实现的ResNet18
    # block = [3, 4, 6, 3]
    # channel = [64, 128, 256, 512]
    # model = ResNetBase(n_blocks=block, n_channels=channel, num_classes=2)  # 自定义实现的ResNet34
    block = [3, 4, 6, 3]
    channel = [256, 512, 1024, 2048]
    bottleneck = [64, 128, 256, 512]
    # 自定义实现的ResNet50
    model = ResNetBase(n_blocks=block, n_channels=channel, n_bottlenecks=bottleneck, num_classes=2)

    '''===========================
        4.构建SummaryWriter
    ==========================='''
    # writer = SummaryWriter('log/resnet18/')  # 记录torchvision实现的ResNet18
    # writer = SummaryWriter('log/resnet34/')  # 记录torchvision实现的ResNet34
    # writer = SummaryWriter('log/resnet50/')  # 记录torchvision实现的ResNet50

    # writer = SummaryWriter('log/ResNet/resnet18/')  # 记录自定义实现的ResNet18
    # writer = SummaryWriter('log/ResNet/resnet34/')  # 记录自定义实现的ResNet34
    writer = SummaryWriter('log/ResNet/resnet50/')  # 记录自定义实现的ResNet50

    '''==============
        5.训练
    =============='''
    # ResNet.to('cuda:0')
    # best_model = train_net(ResNet, train_loader, test_loader, epochs=100, device='cuda:0', writer=writer,
    #                        weights_path='weights/resnet18/')  # 训练torchvision实现的ResNet18
    # best_model = train_net(ResNet, train_loader, test_loader, epochs=100, device='cuda:0', writer=writer,
    #                        weights_path='weights/resnet34/')  # 训练torchvision实现的ResNet34
    # best_model = train_net(ResNet, train_loader, test_loader, epochs=100, device='cuda:0', writer=writer,
    #                        weights_path='weights/resnet50/')  # 训练torchvision实现的ResNet50

    model.to('cuda:0')
    # best_model = train_net(model, train_loader, test_loader, epochs=100, device='cuda:0', writer=writer,
    #                        weights_path='weights/ResNet/resnet18/')  # 训练自定义实现的ResNet18
    # best_model = train_net(model, train_loader, test_loader, epochs=100, device='cuda:0', writer=writer,
    #                        weights_path='weights/ResNet/resnet34/')  # 训练自定义实现的ResNet34
    best_model = train_net(model, train_loader, test_loader, epochs=100, device='cuda:0', writer=writer,
                           weights_path='weights/ResNet/resnet50/')  # 训练自定义实现的ResNet50

    print(best_model)
    # with open('log/resnet18/best_model.json', 'w') as f:
    #     f.write(json.dumps(best_model, indent=4))  # 保存torchvision实现的ResNet18的训练结果
    # with open('log/resnet34/best_model.json', 'w') as f:
    #     f.write(json.dumps(best_model, indent=4))  # 保存torchvision实现的ResNet34的训练结果
    # with open('log/resnet50/best_model.json', 'w') as f:
    #     f.write(json.dumps(best_model, indent=4))  # 保存torchvision实现的ResNet50的训练结果

    # with open('log/ResNet/resnet18/best_model.json', 'w') as f:
    #     f.write(json.dumps(best_model, indent=4))  # 保存自定义实现的ResNet18的训练结果
    # with open('log/ResNet/resnet34/best_model.json', 'w') as f:
    #     f.write(json.dumps(best_model, indent=4))  # 保存自定义实现的ResNet34的训练结果
    with open('log/ResNet/resnet50/best_model.json', 'w') as f:
        f.write(json.dumps(best_model, indent=4))  # 保存自定义实现的ResNet50的训练结果
