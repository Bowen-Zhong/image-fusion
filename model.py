import numpy as np
import torch
import torch.nn as nn
from torchprofile import profile_macs


from timm.models.layers import DropPath
import antialiased_cnns


from Module.SSM_deepthconv import ConvSSM
from Module.WTConv2 import DepthwiseSeparableConvWithWTConv2d




class W_HSSF(nn.Module):
    def __init__(self, patch_size=16, dim=256, num_heads=8, channels=[112, 160, 208, 256],
                 fusionblock_depth=[4, 4, 4, 4], qk_scale=None, attn_drop=0., proj_drop=0.):
        super(W_HSSF, self).__init__()

        self.encoder = encoder_convblock()

        self.conv_up4 = ConvBlock_up(channels[-1] * 2, 104, 208)  # ori
        self.conv_up3 = ConvBlock_up(channels[-2] * 3, 80, 160)  # ori
        self.conv_up2 = ConvBlock_up(channels[-3] * 3, 56, 112)  # ori
        self.conv_up1 = ConvBlock_up(channels[-4] * 3, 8, 16, if_up=False)  # ori

        # Fusion Block
        # self.fusionnet1 = FusionModule(patch_size=patch_size, dim=dim, num_heads=num_heads,
        #                                channels=channels[0],
        #                                fusionblock_depth=fusionblock_depth[0],
        #                                qk_scale=qk_scale, attn_drop=attn_drop,
        #                                proj_drop=proj_drop)
        # self.fusionnet2 = FusionModule(patch_size=patch_size, dim=dim, num_heads=num_heads,
        #                                channels=channels[1],
        #                                fusionblock_depth=fusionblock_depth[1],
        #                                qk_scale=qk_scale, attn_drop=attn_drop,
        #                                proj_drop=proj_drop)
        # self.fusionnet3 = FusionModule(patch_size=patch_size, dim=dim, num_heads=num_heads,
        #                                channels=channels[2],
        #                                fusionblock_depth=fusionblock_depth[2],
        #                                qk_scale=qk_scale, attn_drop=attn_drop,
        #                                proj_drop=proj_drop)
        # self.fusionnet4 = FusionModule(patch_size=patch_size, dim=dim, num_heads=num_heads,
        #                                channels=channels[3],
        #                                fusionblock_depth=fusionblock_depth[3],
        #                                qk_scale=qk_scale, attn_drop=attn_drop,
        #                                proj_drop=proj_drop)

        # Conv 1x1
        self.outlayer = nn.Conv2d(16, 1, 1)
        self.SSM1 = ConvSSM(256)
        self.SSM2 = ConvSSM(128)
        self.SSM3 = ConvSSM(64)
        self.SSM4 = ConvSSM(32)
        # self.SSM1 = ConvSSM(128)
        # self.SSM2 = ConvSSM(64)
        # self.SSM3 = ConvSSM(32)
        # self.SSM4 = ConvSSM(16)

    def forward(self, img1, img2):
        x1, x2, x3, x4 = self.encoder(img1)
        y1, y2, y3, y4 = self.encoder(img2)


        z1 = torch.cat((x1, y1), dim=1)
        z2 = torch.cat((x2, y2), dim=1)
        z3 = torch.cat((x3, y3), dim=1)
        z4 = torch.cat((x4, y4), dim=1)

        z1 = self.SSM1(z1)
        z2 = self.SSM2(z2)
        z3 = self.SSM3(z3)
        z4 = self.SSM4(z4)

        out4 = self.conv_up4(z4)
        out3 = self.conv_up3(torch.cat((out4, z3), dim=1))
        out2 = self.conv_up2(torch.cat((out3, z2), dim=1))
        out1 = self.conv_up1(torch.cat((out2, z1), dim=1))

        img_fusion = self.outlayer(out1)

        return img_fusion




class Conv_decoder(nn.Module):
    def __init__(self, channels=[256, 128, 64, 1]):
        super(Conv_decoder, self).__init__()
        self.decoder1 = Conv_Block(channels[0], int(channels[0] + channels[1] / 2), channels[1])
        self.decoder2 = Conv_Block(channels[1], int(channels[1] + channels[2] / 2), channels[2])
        self.decoder3 = Conv_Block(channels[2], int(channels[2] / 2), channels[3])

    def forward(self, x):
        x1 = self.decoder1(x)
        x2 = self.decoder2(x1)
        out = self.decoder3(x2)

        return out


class Conv_Block(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hid_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(hid_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.norm1 = nn.BatchNorm2d(hid_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.norm1(x1)
        x2 = self.conv2(x1)
        x2 = self.norm2(x2)
        out = self.act(x2)

        return out


class ConvBlock_down(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, kernel_size=3, stride=1, padding=1, if_down=True):
        super(ConvBlock_down, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hid_channels, kernel_size, stride, padding)
        #self.conv2 = nn.Conv2d(hid_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = DepthwiseSeparableConvWithWTConv2d(hid_channels, out_channels)

        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.GELU()

        self.if_down = if_down
        self.down = nn.MaxPool2d(kernel_size=2, stride=1)
        self.down_anti = antialiased_cnns.BlurPool(in_channels, stride=2)

    def forward(self, x):
        if self.if_down:
            x = self.down(x)
            x = self.down_anti(x)
            x = self.act(x)

        x1 = self.conv1(x)
        x2 = self.bn1(x1)

        x3 = self.conv2(x2)
        x3 = self.bn2(x3)
        out = self.act(x3)

        return out


class encoder_convblock(nn.Module):
    def __init__(self):
        super(encoder_convblock, self).__init__()
        self.inlayer = nn.Conv2d(1, 64, 1)
        self.block1 = ConvBlock_down(64, 32, 112, if_down=False)
        self.block2 = ConvBlock_down(112, 56, 160)
        self.block3 = ConvBlock_down(160, 80, 208)
        self.block4 = ConvBlock_down(208, 104, 256)

    def forward(self, img):
        img = self.inlayer(img)
        x1 = self.block1(img)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        return x1, x2, x3, x4


class ConvBlock_up(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, kernel_size=3, stride=1, padding=1, if_up=True):
        super(ConvBlock_up, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hid_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(hid_channels, out_channels, kernel_size, stride, padding)

        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.GELU()

        self.if_up = if_up
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.bn1(x1)

        x3 = self.conv2(x2)
        x3 = self.bn2(x3)

        if self.if_up:
            out = self.act(self.up(x3))
        else:
            out = self.act(x3)

        return out






if __name__ == '__main__':
    device = 'cuda:0'
    img1 = torch.randn(1, 1, 224, 224).to(device)
    img2 = torch.randn(1, 1, 224, 224).to(device)
    model = W_HSSF().to(device)
    result = model(img1, img2)
    print(result.shape)

    # model = NestFuse()
    # inputs = torch.randn(8, 1, 256, 256)
     #encode = model.encoder(inputs)

    # print(encode[3].size())
    # outputs = model.decoder_train(encode)
    # print(outputs[0].size())
    flops = profile_macs(model, (img1, img2))
    print(flops/1e9)
    params = sum(p.numel() for p in model.parameters())
    print(params/1e6)