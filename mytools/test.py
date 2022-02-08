#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2

    center = kernel_size / 2
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)
#
# x = plt.imread(r"D:\360MoveData\Users\xuhuan\Desktop\pick_part\248105.jpg")
# print(x.shape)
# x = torch.from_numpy(x.astype('float32')).permute(2, 0, 1).unsqueeze(0)
# conv_trans = nn.ConvTranspose2d(3, 3, 4, 2, 1)
# # 将其定义为 bilinear kernel
# conv_trans.weight.data = bilinear_kernel(3, 3, 4)
# y = conv_trans(x).data.squeeze().permute(1, 2, 0).numpy()
# plt.imshow(y.astype('uint8'))
# plt.show()
# print(y.shape)

#************************************Resnet101-8s******************************************
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(ResBlock,self).__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_chans, out_chans):
        super(Bottleneck,self).__init__()
        assert out_chans % 4 == 0
        self.block1 = ResBlock(in_chans, int(out_chans/4), kernel_size=1, stride=1, padding=0)
        self.block2 = ResBlock(int(out_chans/4), int(out_chans/4), kernel_size=3, stride=1, padding=1)
        self.block3 = ResBlock(int(out_chans/4), out_chans, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out += identity
        return out


class DownBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_chans, out_chans, stride=2):
        super(DownBottleneck,self).__init__()
        assert out_chans % 4 == 0
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=stride, padding=0)
        self.block1 = ResBlock(in_chans, int(out_chans/4), kernel_size=1, stride=stride, padding=0)
        self.block2 = ResBlock(int(out_chans/4), int(out_chans/4), kernel_size=3, padding=1)
        self.block3 = ResBlock(int(out_chans/4), out_chans, kernel_size=1, padding=0)
    def forward(self, x):
        identity = self.conv1(x)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out += identity
        return out


def make_layers(in_channels, layer_list, name="vgg"):
    layers = []
    if name == "vgg":
        for v in layer_list:
            layers += [Block(in_channels, v)]
            in_channels = v
    elif name == "resnet":
        layers += [DownBottleneck(in_channels, layer_list[0])]
        in_channels = layer_list[0]
        for v in layer_list[1:]:
            layers += [Bottleneck(in_channels, v)]
            in_channels = v
    return nn.Sequential(*layers)


class Layer(nn.Module):
    def __init__(self, in_channels, layer_list, net_name):
        super(Layer, self).__init__()
        self.layer = make_layers(in_channels, layer_list, name=net_name)

    def forward(self, x):
        out = self.layer(x)
        return out


class ResNet101(nn.Module):
    '''
    ResNet101 model
    '''
    def __init__(self):
        super(ResNet101, self).__init__()
        self.conv1 = Block(3, 64, 7, 2, 3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2_1 = DownBottleneck(64, 256, stride=1)
        self.conv2_2 = Bottleneck(256, 256)
        self.conv2_3 = Bottleneck(256, 256)
        self.layer3 = Layer(256, [512]*2, "resnet")
        self.layer4 = Layer(512, [1024]*23, "resnet")
        self.layer5 = Layer(1024, [2048]*3, "resnet")
    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(self.pool1(f1))))
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        f5 = self.layer5(f4)
        return [f2, f3, f4, f5]


class Resnet101_8s(nn.Module):
    def __init__(self, n_class):
        super(Resnet101_8s, self).__init__()
        self.encode = ResNet101()

        self.score_fr = nn.Conv2d(2048, n_class, 1)
        self.trans_p4 = nn.Conv2d(1024, n_class, 1)
        self.trans_p3 = nn.Conv2d(512, n_class, 1)
        self.smooth_conv1 = nn.Conv2d(n_class, n_class, 3, padding=1)
        self.smooth_conv2 = nn.Conv2d(n_class, n_class, 3, padding=1)

        self.up2time = nn.ConvTranspose2d(
            n_class, n_class, 2, stride=2, bias=False)
        self.up4time = nn.ConvTranspose2d(
            n_class, n_class, 2, stride=2, bias=False)
        self.up32time = nn.ConvTranspose2d(
            n_class, n_class, 8, stride=8, bias=False)

    def forward(self, x):
        feature_list = self.encode(x)
        p2, p3, p4, p5 = feature_list

        f7 = self.score_fr(p5)
        up2_feat = self.up2time(f7)

        h = self.trans_p4(p4)
        h = h + up2_feat
        h = self.smooth_conv1(h)

        up4_feat = self.up4time(h)
        h = self.trans_p3(p3)
        h = h + up4_feat
        h = self.smooth_conv2(h)

        final_scores = self.up32time(h)
        return final_scores

# x = torch.randn(1,3, 256,256)
# fcn_resnet = Resnet101_8s(3)
# fcn_resnet.eval()
# y_resnet = fcn_resnet(x)

#************************************UNet******************************************
class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))

        block.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_chans, out_chans, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class UNet(nn.Module):
    def __init__(self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=True,
        up_mode='upconv',
    ):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)

# x = torch.randn((1,3, 572,572))
# unet = UNet(3)
# unet.eval()
# y_unet = unet(x)

#**************************************ResNetUNet****************************
class ResNetUNet(nn.Module):
    def __init__(
        self,
        n_classes=2,
        depth=5,
        wf=6,
        padding=1,
        batch_norm=False,
        up_mode='upconv',
    ):
        super(ResNetUNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = 2 ** (wf + depth)
        self.encode = ResNet101()
        self.up_path = nn.ModuleList()
        for i in reversed(range(2,depth)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = self.encode(x)
        x = blocks[-1]
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 2])

        return self.last(x)


# x = torch.randn((1,3,256,256))
# unet = ResNetUNet()
# unet.eval()
# y_unet = unet(x)

# from torchsummary import summary
# model = UNet(3)
# summary(model, (3,256,256),device="cpu")

#  **************************Xception************************
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=0):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, k, s, p, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class SepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=0):
        super(SeparableConv2d, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sepconv = SeparableConv2d(in_channels, out_channels, k=k, s=s, p=p)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.bn(self.sepconv(self.relu(x)))
        return out


class Block_X(nn.Module):
    def __init__(self, channels):
        super(Block_X, self).__init__()

        self.sepconv1 = SepConvBlock(channels, channels, k=3, s=1, p=1)
        self.sepconv2 = SepConvBlock(channels, channels, k=3, s=1, p=1)
        self.sepconv3 = SepConvBlock(channels, channels, k=3, s=1, p=1)

    def forward(self, x):
        identify = x
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        out = self.sepconv3(x)
        out += x
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()

        self.sepconv1 = SepConvBlock(in_channels, out_channels, k=3, s=1, p=1)
        self.sepconv2 = SepConvBlock(out_channels, out_channels, k=3, s=1, p=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=2)

    def forward(self, x):
        identify = self.skip(x)
        out = self.pool(self.sepconv2(self.sepconv1(x)))
        out += identify
        return out


class Xception(nn.Module):

    def __init__(self, num_classes=10):
        super(Xception, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = DownBlock(64, 128)
        self.block2 = DownBlock(128, 256)
        self.block3 = DownBlock(256, 728)

        self.block4 = Block_X(728)
        self.block5 = Block_X(728)
        self.block6 = Block_X(728)
        self.block7 = Block_X(728)
        self.block8 = Block_X(728)
        self.block9 = Block_X(728)
        self.block10 = Block_X(728)
        self.block11 = Block_X(728)

        self.sepconv12 = SepConvBlock(728, 728, k=3, s=1, p=1)
        self.sepconv13 = SepConvBlock(728, 1024, k=3, s=1, p=1)
        self.pool14 = nn.MaxPool2d(3, stride=2, padding=1)
        self.skip15 = nn.Conv2d(728, 1024, 1, stride=2)

        self.conv16 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn16 = nn.BatchNorm2d(1536)
        self.relu16 = nn.ReLU(inplace=True)

        # do relu here
        self.conv17 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn17 = nn.BatchNorm2d(2048)
        self.relu17 = nn.ReLU(inplace=True)

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))

        x = self.bn2(self.conv2(x))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        out = self.pool14(self.sepconv13(self.sepconv12(x)))
        skip = self.skip15(x)
        x = out + skip

        x = self.relu16(self.bn16(self.conv16(x)))
        x = self.relu17(self.bn17(self.conv17(x)))

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


#***********************************mobilenetV2****************************************
import torch.nn as nn
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2(pretrained=True):
    model = MobileNetV2(width_mult=1)

    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    net = mobilenet_v2(True)