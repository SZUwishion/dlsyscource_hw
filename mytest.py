import sys
sys.path.append('./python')
import torch
import needle.nn as nn
import needle as ndl
import numpy as np

class torch_convbn(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(torch_convbn, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dtype=torch.float32, bias=True)
        self.bn = torch.nn.BatchNorm2d(out_channels, dtype=torch.float32)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

def test_convbn():
    # 创建随机输入张量
    a = nn.init.rand(2, 3, 4, 4, device=ndl.cpu())
    torch_a = torch.tensor(a.cached_data.numpy(), requires_grad=True)

    # 创建 needle 的 ConvBN 层
    ndl_conv_bn = nn.ConvBN(in_channels=3, out_channels=16, kernel_size=3, stride=1, device=None)

    # 创建 torch 的 Conv2d 和 BatchNorm2d 层
    torch_conv = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
    torch_bn = torch.nn.BatchNorm2d(16)
    torch_relu = torch.nn.ReLU()

    # 将 needle 的权重复制到 torch
    with torch.no_grad():
        torch_conv.weight.copy_(torch.tensor(ndl_conv_bn.conv.weight.cached_data.numpy().transpose(3, 2, 0, 1)))
        torch_bn.weight.copy_(torch.tensor(ndl_conv_bn.bn.weight.cached_data.numpy()))

    # 前向传播
    ndl_output = ndl_conv_bn(a)
    torch_output = torch_relu(torch_bn(torch_conv(torch_a)))

    # 打印输出
    print("Needle ConvBN output shape:", ndl_output.shape)
    print("Torch ConvBN output shape:", torch_output.shape)
    print("Difference (L2 norm):", np.linalg.norm(ndl_output.cached_data.numpy() - torch_output.detach().numpy()))


def test_linear():
    # 创建随机输入张量
    ndl_input = ndl.init.rand(10, 5, device=ndl.cpu())
    torch_input = torch.tensor(ndl_input.cached_data.numpy(), requires_grad=True)

    # 创建 needle 的 Linear 层
    ndl_linear = nn.Linear(in_features=5, out_features=3, device=ndl.cpu())

    # 创建 torch 的 Linear 层
    torch_linear = torch.nn.Linear(in_features=5, out_features=3, bias=True)

    # 将 needle 的权重和偏置复制到 torch
    with torch.no_grad():
        torch_linear.weight.copy_(torch.tensor(ndl_linear.weight.cached_data.numpy().transpose()))
        torch_linear.bias.copy_(torch.tensor(ndl_linear.bias.cached_data.numpy().reshape(-1)))

    # 前向传播
    ndl_output = ndl_linear(ndl_input)
    torch_output = torch_linear(torch_input)

    # 打印输出
    print("Difference (L2 norm):", np.linalg.norm(ndl_output.cached_data.numpy() - torch_output.detach().numpy()))


def test_res():
    # 创建随机输入张量
    a = nn.init.rand(2, 3, 4, 4, device=ndl.cpu())
    torch_a = torch.tensor(a.cached_data.numpy(), requires_grad=True)

    # 创建 needle 的 Residual 层
    ndl_convbn1 = nn.ConvBN(in_channels=3, out_channels=16, kernel_size=3, stride=1, device=ndl.cpu())
    ndl_convbn2 = nn.ConvBN(in_channels=16, out_channels=3, kernel_size=3, stride=1, device=ndl.cpu())
    res = nn.Residual(nn.Sequential(ndl_convbn1, ndl_convbn2))

    # 创建 torch 的 Residual 层
    torch_conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
    torch_bn1 = torch.nn.BatchNorm2d(16)
    torch_relu = torch.nn.ReLU()
    torch_conv2 = torch.nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
    torch_bn2 = torch.nn.BatchNorm2d(3)

    # 将 needle 的权重复制到 torch
    with torch.no_grad():
        torch_conv1.weight.copy_(torch.tensor(ndl_convbn1.conv.weight.cached_data.numpy().transpose(3, 2, 0, 1)))
        torch_bn1.weight.copy_(torch.tensor(ndl_convbn1.bn.weight.cached_data.numpy()))
        torch_conv2.weight.copy_(torch.tensor(ndl_convbn2.conv.weight.cached_data.numpy().transpose(3, 2, 0, 1)))
        torch_bn2.weight.copy_(torch.tensor(ndl_convbn2.bn.weight.cached_data.numpy()))

    # 前向传播
    ndl_output = res(a)
    torch_output = torch_relu(torch_bn2(torch_conv2(torch_relu(torch_bn1(torch_conv1(torch_a)))))) + torch_a

    # 打印输出
    print("Needle Residual output shape:", ndl_output.shape)
    print("Torch Residual output shape:", torch_output.shape)
    print("Difference (L2 norm):", np.linalg.norm(ndl_output.cached_data.numpy() - torch_output.detach().numpy()))

def test_relu():
    # 创建随机输入张量
    a = nn.init.rand(2, 3, 4, 4, device=ndl.cpu())
    torch_a = torch.tensor(a.cached_data.numpy(), requires_grad=True)

    # 创建 needle 的 ReLU 层
    ndl_relu = nn.ReLU()

    # 创建 torch 的 ReLU 层
    torch_relu = torch.nn.ReLU()

    # 前向传播
    ndl_output = ndl_relu(a)
    torch_output = torch_relu(torch_a)

    # 打印输出
    print("Difference (L2 norm):", np.linalg.norm(ndl_output.cached_data.numpy() - torch_output.detach().numpy()))


def test_resnet9():
    # 创建随机输入张量
    a = nn.init.rand(2, 3, 32, 32, device=ndl.cpu())
    torch_a = torch.tensor(a.cached_data.numpy(), requires_grad=True)

    convbn1 = nn.ConvBN(in_channels=3, out_channels=16, kernel_size=7, stride=4, device=ndl.cpu())
    torch_convbn1 = torch_convbn(3, 16, 7, 4, 3)
    with torch.no_grad():
        torch_convbn1.conv.weight.copy_(torch.tensor(convbn1.conv.weight.cached_data.numpy().transpose(3, 2, 0, 1)))
        torch_convbn1.conv.bias.copy_(torch.tensor(convbn1.conv.bias.cached_data.numpy()))
        torch_convbn1.bn.weight.copy_(torch.tensor(convbn1.bn.weight.cached_data.numpy()))
    out1 = convbn1(a)
    torch_out1 = torch_convbn1(torch_a)
    print("Difference (L2 norm) for convbn1:", np.linalg.norm(out1.cached_data.numpy() - torch_out1.detach().numpy()))

    convbn2 = nn.ConvBN(in_channels=16, out_channels=32, kernel_size=3, stride=2, device=ndl.cpu())
    torch_convbn2 = torch_convbn(16, 32, 3, 2, 1)
    with torch.no_grad():
        torch_convbn2.conv.weight.copy_(torch.tensor(convbn2.conv.weight.cached_data.numpy().transpose(3, 2, 0, 1)))
        torch_convbn2.conv.bias.copy_(torch.tensor(convbn2.conv.bias.cached_data.numpy()))
        torch_convbn2.bn.weight.copy_(torch.tensor(convbn2.bn.weight.cached_data.numpy()))
    out2 = convbn2(out1)
    torch_out2 = torch_convbn2(torch_out1)
    print("Difference (L2 norm) for convbn2:", np.linalg.norm(out2.cached_data.numpy() - torch_out2.detach().numpy()))

    convbn3 = nn.ConvBN(in_channels=32, out_channels=32, kernel_size=3, stride=1, device=ndl.cpu())
    torch_convbn3 = torch_convbn(32, 32, 3, 1, 1)
    with torch.no_grad():
        torch_convbn3.conv.weight.copy_(torch.tensor(convbn3.conv.weight.cached_data.numpy().transpose(3, 2, 0, 1)))
        torch_convbn3.conv.bias.copy_(torch.tensor(convbn3.conv.bias.cached_data.numpy()))
        torch_convbn3.bn.weight.copy_(torch.tensor(convbn3.bn.weight.cached_data.numpy()))
    out3 = convbn3(out2)
    torch_out3 = torch_convbn3(torch_out2)

    convbn4 = nn.ConvBN(in_channels=32, out_channels=32, kernel_size=3, stride=1, device=ndl.cpu())
    torch_convbn4 = torch_convbn(32, 32, 3, 1, 1)
    with torch.no_grad():
        torch_convbn4.conv.weight.copy_(torch.tensor(convbn4.conv.weight.cached_data.numpy().transpose(3, 2, 0, 1)))
        torch_convbn4.conv.bias.copy_(torch.tensor(convbn4.conv.bias.cached_data.numpy()))
        torch_convbn4.bn.weight.copy_(torch.tensor(convbn4.bn.weight.cached_data.numpy()))
    out4 = convbn4(out3)
    torch_out4 = torch_convbn4(torch_out3)

    res1 = nn.Residual(nn.Sequential(convbn3, convbn4))
    out5 = res1(out2)
    torch_out5 = torch_out2 + torch_out4
    print("Difference (L2 norm) for res1:", np.linalg.norm(out5.cached_data.numpy() - torch_out5.detach().numpy()))

    convbn5 = nn.ConvBN(in_channels=32, out_channels=64, kernel_size=3, stride=2, device=ndl.cpu())
    torch_convbn5 = torch_convbn(32, 64, 3, 2, 1)
    with torch.no_grad():
        torch_convbn5.conv.weight.copy_(torch.tensor(convbn5.conv.weight.cached_data.numpy().transpose(3, 2, 0, 1)))
        torch_convbn5.conv.bias.copy_(torch.tensor(convbn5.conv.bias.cached_data.numpy()))
        torch_convbn5.bn.weight.copy_(torch.tensor(convbn5.bn.weight.cached_data.numpy()))
    out6 = convbn5(out5)
    torch_out6 = torch_convbn5(torch_out5)

    convbn6 = nn.ConvBN(in_channels=64, out_channels=128, kernel_size=3, stride=2, device=ndl.cpu())
    torch_convbn6 = torch_convbn(64, 128, 3, 2, 1)
    with torch.no_grad():
        torch_convbn6.conv.weight.copy_(torch.tensor(convbn6.conv.weight.cached_data.numpy().transpose(3, 2, 0, 1)))
        torch_convbn6.conv.bias.copy_(torch.tensor(convbn6.conv.bias.cached_data.numpy()))
        torch_convbn6.bn.weight.copy_(torch.tensor(convbn6.bn.weight.cached_data.numpy()))
    out7 = convbn6(out6)
    torch_out7 = torch_convbn6(torch_out6)

    convbn7 = nn.ConvBN(in_channels=128, out_channels=128, kernel_size=3, stride=1, device=ndl.cpu())
    torch_convbn7 = torch_convbn(128, 128, 3, 1, 1)
    with torch.no_grad():
        torch_convbn7.conv.weight.copy_(torch.tensor(convbn7.conv.weight.cached_data.numpy().transpose(3, 2, 0, 1)))
        torch_convbn7.conv.bias.copy_(torch.tensor(convbn7.conv.bias.cached_data.numpy()))
        torch_convbn7.bn.weight.copy_(torch.tensor(convbn7.bn.weight.cached_data.numpy()))
    out8 = convbn7(out7)
    torch_out8 = torch_convbn7(torch_out7)

    convbn8 = nn.ConvBN(in_channels=128, out_channels=128, kernel_size=3, stride=1, device=ndl.cpu())
    torch_convbn8 = torch_convbn(128, 128, 3, 1, 1)
    with torch.no_grad():
        torch_convbn8.conv.weight.copy_(torch.tensor(convbn8.conv.weight.cached_data.numpy().transpose(3, 2, 0, 1)))
        torch_convbn8.conv.bias.copy_(torch.tensor(convbn8.conv.bias.cached_data.numpy()))
        torch_convbn8.bn.weight.copy_(torch.tensor(convbn8.bn.weight.cached_data.numpy()))
    out9 = convbn8(out8)
    torch_out9 = torch_convbn8(torch_out8)

    res2 = nn.Residual(nn.Sequential(convbn7, convbn8))
    out10 = res2(out7)
    torch_out10 = torch_out7 + torch_out9
    print("Difference (L2 norm) for res2:", np.linalg.norm(out10.cached_data.numpy() - torch_out10.detach().numpy()))

    flatten = nn.Flatten()
    torch_flatten = torch.nn.Flatten()
    out11 = flatten(out10)
    torch_out11 = torch_flatten(torch_out10)
    print("Difference (L2 norm) for flatten:", np.linalg.norm(out11.cached_data.numpy() - torch_out11.detach().numpy()))

    linear1 = nn.Linear(in_features=128, out_features=128, device=ndl.cpu())
    torch_linear1 = torch.nn.Linear(128, 128)
    with torch.no_grad():
        torch_linear1.weight.copy_(torch.tensor(linear1.weight.cached_data.numpy().transpose()))
        torch_linear1.bias.copy_(torch.tensor(linear1.bias.cached_data.numpy().reshape(-1)))
    out12 = linear1(out11)
    torch_out12 = torch_linear1(torch_out11)
    print("Difference (L2 norm) for linear1:", np.linalg.norm(out12.cached_data.numpy() - torch_out12.detach().numpy()))

    relu = nn.ReLU()
    torch_relu = torch.nn.ReLU()
    out13 = relu(out12)
    torch_out13 = torch_relu(torch_out12)
    print("Difference (L2 norm) for relu:", np.linalg.norm(out13.cached_data.numpy() - torch_out13.detach().numpy()))

    linear2 = nn.Linear(in_features=128, out_features=10, device=ndl.cpu())
    torch_linear2 = torch.nn.Linear(128, 10)
    with torch.no_grad():
        torch_linear2.weight.copy_(torch.tensor(linear2.weight.cached_data.numpy().transpose()))
        torch_linear2.bias.copy_(torch.tensor(linear2.bias.cached_data.numpy().reshape(-1)))
    out14 = linear2(out13)
    torch_out14 = torch_linear2(torch_out13)
    print("Difference (L2 norm) for linear2:", np.linalg.norm(out14.cached_data.numpy() - torch_out14.detach().numpy()))

    print(out14)
    print(torch_out14)


test_resnet9()