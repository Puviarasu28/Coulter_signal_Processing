import torch
import torch.nn as nn

#(kernel size, filter num, stride, padding), "M" - Max-pool layer
architecture_1 = [(3, 32, 1),
                  "M",
                  (3, 64, 1),
                  "M",
                  (3, 128, 1),
                  (1, 64, 1),
                  (3, 128, 1),
                  "M",
                  (3, 256, 1),
                  (1, 128, 1),
                  (3, 256, 1),
                  "M",
                  (3, 512, 1),
                  (1, 256, 1),
                  (3, 512, 1),
                  (1, 256, 1),
                  (3, 512, 1),
                  ]

architecture_2 = ["M",
                  (3, 1024, 1),
                  (1, 512, 1),
                  (3, 1024, 1),
                  (1, 512, 1),
                  (3, 1024, 1),
                  (3, 1024, 1),
                  (3, 1024, 1),
                  ]

architecture_3 = [(1, 64, 1)]

architecture_4 = [(3, 1024, 1)]

architecture_5 = [[1, 420, 1]]


class CNNBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class CNNBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock2, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        return self.conv(x)


class LambdaLayer(nn.Module):
    def __init__(self):
        super(LambdaLayer, self).__init__()
        self.block_size = 2

    def forward(self, x):
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, self.block_size, stride=self.block_size)
        return unfolded_x.view(n, c * self.block_size * self.block_size, h // self.block_size, w // self.block_size)


class Yolov2(nn.Module):
    def __init__(self, in_channels=1, **kwargs):
        super(Yolov2, self).__init__()
        self.darknet1 = self._create_conv_layers(architecture_1,1)
        self.darknet2 = self._create_conv_layers(architecture_2,512)
        self.darknet3 = self._create_conv_layers(architecture_3,512)
        self.darknet4 = self._create_conv_layers(architecture_4,1152)
        self.darknet5 = self._create_conv_layers(architecture_5,1024)

    def forward(self, x):
        H = 13
        W = 13
        C = 80
        B = 5
        x = self.darknet1(x)
        skip_connection = x
        x = self.darknet2(x)
        skip_connection = self.darknet3(skip_connection)
        lambda1 = LambdaLayer()
        skip_connection = lambda1(skip_connection)
        x = torch.cat([skip_connection, x], 1)
        x = self.darknet4(x)
        x = self.darknet5(x)
        x = x.reshape(-1, H, W, B, 4 + C)
        return x

    def _create_conv_layers(self, architecture,no_channels):
        layers = []
        in_channels = no_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock1(in_channels, out_channels=x[1],
                                    kernel_size=x[0], stride=x[2], padding='same')]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2)]
            elif type(x) == list:
                layers += [CNNBlock2(in_channels, out_channels=x[1],
                                     kernel_size=x[0], stride=x[2], padding='same')]
                in_channels = x[1]
        return nn.Sequential(*layers)