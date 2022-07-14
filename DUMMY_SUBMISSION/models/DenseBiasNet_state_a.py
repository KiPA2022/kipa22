from torch import nn, cat, add
import torch.nn.functional as F


class conv_bias(nn.Module):
    def __init__(self, in_ch, out_ch, bias_size=1):
        super(conv_bias, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.bias_size = bias_size

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        x_bias = x[:, 0:self.bias_size, :, :, :]
        return x_bias, x

class DenseBiasNet_base(nn.Module):
    def __init__(self, n_channels, depth=(16, 16, 32, 32, 64, 64, 128, 128, 64, 64, 32, 32, 16, 16), bias=1):
        super(DenseBiasNet_base, self).__init__()
        self.depth = depth
        self.conv0 = conv_bias(n_channels, depth[0], bias_size=bias)

        self.conv1 = conv_bias(depth[0], depth[1], bias_size=bias)

        in_chan = bias
        self.conv2 = conv_bias(depth[1] + in_chan, depth[2], bias_size=bias)

        in_chan = in_chan + bias
        self.conv3 = conv_bias(depth[2] + in_chan, depth[3], bias_size=bias)

        in_chan = in_chan + bias
        self.conv4 = conv_bias(depth[3] + in_chan, depth[4], bias_size=bias)

        in_chan = in_chan + bias
        self.conv5 = conv_bias(depth[4] + in_chan, depth[5], bias_size=bias)

        in_chan = in_chan + bias
        self.conv6 = conv_bias(depth[5] + in_chan, depth[6], bias_size=bias)

        in_chan = in_chan + bias
        self.conv7 = conv_bias(depth[6] + in_chan, depth[7], bias_size=bias)

        in_chan = in_chan + bias
        self.conv8 = conv_bias(depth[7] + in_chan, depth[8], bias_size=bias)

        in_chan = in_chan + bias
        self.conv9 = conv_bias(depth[8] + in_chan, depth[9], bias_size=bias)

        in_chan = in_chan + bias
        self.conv10 = conv_bias(depth[9] + in_chan, depth[10], bias_size=bias)

        in_chan = in_chan + bias
        self.conv11 = conv_bias(depth[10] + in_chan, depth[11], bias_size=bias)

        in_chan = in_chan + bias
        self.conv12 = conv_bias(depth[11] + in_chan, depth[12], bias_size=bias)

        in_chan = in_chan + bias
        self.conv13 = conv_bias(depth[12] + in_chan, depth[13], bias_size=bias)

        self.maxpooling = nn.MaxPool3d(2)

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
    def forward(self, x):
        # block0
        x_bias_0_0, x = self.conv0(x)
        x_bias_0_1, x_fe0 = self.conv1(x)

        x_bias_0_0_1 = self.maxpooling(x_bias_0_0)
        x_bias_0_0_2 = self.maxpooling(x_bias_0_0_1)
        x_bias_0_0_3 = self.maxpooling(x_bias_0_0_2)

        x_bias_0_1_1 = self.maxpooling(x_bias_0_1)
        x_bias_0_1_2 = self.maxpooling(x_bias_0_1_1)
        x_bias_0_1_3 = self.maxpooling(x_bias_0_1_2)

        # block1
        x = self.maxpooling(x_fe0)
        x_bias_1_0, x = self.conv2(cat([x, x_bias_0_0_1], dim=1))
        x_bias_1_1, x_fe1 = self.conv3(cat([x, x_bias_0_0_1, x_bias_0_1_1], dim=1))

        x_bias_1_0_0 = self.up(x_bias_1_0)
        x_bias_1_0_2 = self.maxpooling(x_bias_1_0)
        x_bias_1_0_3 = self.maxpooling(x_bias_1_0_2)

        x_bias_1_1_0 = self.up(x_bias_1_1)
        x_bias_1_1_2 = self.maxpooling(x_bias_1_1)
        x_bias_1_1_3 = self.maxpooling(x_bias_1_1_2)

        # block2
        x = self.maxpooling(x_fe1)
        x_bias_2_0, x = self.conv4(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2], dim=1))
        x_bias_2_1, x_fe2 = self.conv5(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2, x_bias_1_1_2], dim=1))

        x_bias_2_0_1 = self.up(x_bias_2_0)
        x_bias_2_0_0 = self.up(x_bias_2_0_1)
        x_bias_2_0_3 = self.maxpooling(x_bias_2_0)

        x_bias_2_1_1 = self.up(x_bias_2_1)
        x_bias_2_1_0 = self.up(x_bias_2_1_1)
        x_bias_2_1_3 = self.maxpooling(x_bias_2_1)

        # block3
        x = self.maxpooling(x_fe2)
        x_bias_3_0, x = self.conv6(
            cat([x, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3], dim=1))
        x_bias_3_1, x_fe3 = self.conv7(cat([x, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3,
                                        x_bias_2_1_3], dim=1))

        x_bias_3_0_2 = self.up(x_bias_3_0)
        x_bias_3_0_1 = self.up(x_bias_3_0_2)
        x_bias_3_0_0 = self.up(x_bias_3_0_1)

        x_bias_3_1_2 = self.up(x_bias_3_1)
        x_bias_3_1_1 = self.up(x_bias_3_1_2)
        x_bias_3_1_0 = self.up(x_bias_3_1_1)

        # block4
        x = self.up(x_fe3)
        x_bias_2_2, x = self.conv8(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2, x_bias_1_1_2, x_bias_2_0,
                                        x_bias_2_1, x_bias_3_0_2], dim=1))
        x_bias_2_3, x_fe4 = self.conv9(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2, x_bias_1_1_2, x_bias_2_0,
                                        x_bias_2_1, x_bias_3_0_2, x_bias_3_1_2], dim=1))

        x_bias_2_2_1 = self.up(x_bias_2_2)
        x_bias_2_2_0 = self.up(x_bias_2_2_1)

        x_bias_2_3_1 = self.up(x_bias_2_3)
        x_bias_2_3_0 = self.up(x_bias_2_3_1)

        # block5
        x = self.up(x_fe4)
        x_bias_1_2, x = self.conv10(cat([x, x_bias_0_0_1, x_bias_0_1_1, x_bias_1_0, x_bias_1_1, x_bias_2_0_1,
                                         x_bias_2_1_1, x_bias_3_0_1, x_bias_3_1_1, x_bias_2_2_1], dim=1))
        x_bias_1_3, x_fe5 = self.conv11(cat([x, x_bias_0_0_1, x_bias_0_1_1, x_bias_1_0, x_bias_1_1, x_bias_2_0_1,
                                         x_bias_2_1_1, x_bias_3_0_1, x_bias_3_1_1, x_bias_2_2_1, x_bias_2_3_1], dim=1))

        x_bias_1_2_0 = self.up(x_bias_1_2)
        x_bias_1_3_0 = self.up(x_bias_1_3)

        # block6
        x = self.up(x_fe5)
        x_bias_0_2, x = self.conv12(cat([x, x_bias_0_0, x_bias_0_1, x_bias_1_0_0, x_bias_1_1_0, x_bias_2_0_0,
                                         x_bias_2_1_0, x_bias_3_0_0, x_bias_3_1_0, x_bias_2_2_0, x_bias_2_3_0,
                                         x_bias_1_2_0], dim=1))

        x_bias_0_3, x_fe6 = self.conv13(cat([x, x_bias_0_0, x_bias_0_1, x_bias_1_0_0, x_bias_1_1_0, x_bias_2_0_0,
                                         x_bias_2_1_0, x_bias_3_0_0, x_bias_3_1_0, x_bias_2_2_0, x_bias_2_3_0,
                                         x_bias_1_2_0, x_bias_1_3_0], dim=1))

        return x_fe6

class DenseBiasNet(nn.Module):
    def __init__(self, n_channels, n_classes, depth=(16, 16, 32, 32, 64, 64, 128, 128, 64, 64, 32, 32, 16, 16), bias=4):
        super(DenseBiasNet, self).__init__()
        self.densebisanet = DenseBiasNet_base(n_channels, depth, bias)
        self.out_conv = nn.Conv3d(depth[-1], n_classes, 3, padding=1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):

        Z = x.size()[2]
        Y = x.size()[3]
        X = x.size()[4]
        diffZ = (8 - Z % 8) % 8
        diffY = (8 - Y % 8) % 8
        diffX = (8 - X % 8) % 8

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])
        x = self.densebisanet(x)
        x = self.out_conv(x)
        x = self.softmax(x)
        return x[:, :, diffZ//2: Z+diffZ//2, diffY//2: Y+diffY//2, diffX // 2:X + diffX // 2]