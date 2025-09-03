"""
Implementation of YOLOv3 architecture
"""
import warnings

import torch
import torch.nn as nn
from timeit import default_timer as timer
from time import perf_counter
""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        #self.norm = nn.GroupNorm(32, out_channels)
        # if bn_act:
        #     self.norm = nn.GroupNorm(32, out_channels)
        # self.leaky = nn.LeakyReLU(0.1)
        self.leaky = nn.SiLU()
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.norm(self.conv(x)))
        else:
            return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
#         self.reg_conv = CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1)
#         self.reg_pred_conv = CNNBlock(2 * in_channels, 4 * 3, bn_act=False, kernel_size=1)
#         self.obj_pred_conv = CNNBlock(2 * in_channels, 1 * 3, bn_act=False, kernel_size=1)
        
#         self.class_conv = CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1)
#         self.class_pred_conv = CNNBlock(2 * in_channels, 3 * 3, bn_act=False, kernel_size=1)
        
#         self.assov_conv = CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1)
#         self.assov_pred_conv = CNNBlock(2 * in_channels, 3 * 3, bn_act=False, kernel_size=1)
        
#         self.state_conv = CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1)
#         self.state_pred_conv = CNNBlock(2 * in_channels, 3 * 3, bn_act=False, kernel_size=1)
        
#         self.pred = nn.Sequential(
#             CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
#             CNNBlock(
#                 2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
#             ),
#         )
        
        width=2
        act = "lrelu"
        self.stem = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                int(in_channels*2),
                int(256*width),
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.cls_conv = nn.Sequential(
            *[
                CNNBlock(
                    256*width,
                    256*width,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                CNNBlock(
                    256*width,
                    256*width,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ]
        )

        self.asso_vec_conv = nn.Sequential(
            *[
                CNNBlock(
                    256*width,
                    256*width,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                CNNBlock(
                    256*width,
                    256*width,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ]
        )

        self.contact_state_conv = nn.Sequential(
            *[
                CNNBlock(
                    256*width,
                    256*width,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                CNNBlock(
                    256*width,
                    256*width,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ]
        )

        self.reg_conv = nn.Sequential(
            *[
                CNNBlock(
                    256*width,
                    256*width,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                CNNBlock(
                    256*width,
                    256*width,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ]
        )

        self.cls_pred = nn.Conv2d(
            in_channels=int(256*width),
            out_channels=3 * 3,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # self.handside_pred = nn.Conv2d(
        #     in_channels=int(256),
        #     out_channels=1 * 3,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        # )

        self.reg_pred = nn.Conv2d(
            in_channels=int(256*width),
            out_channels=4 * 3,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.contact_state_pred = nn.Conv2d(
            in_channels=int(256*width),
            out_channels=1*3,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.association_vector_unitdxy_pred = nn.Conv2d(
            in_channels=int(256*width),
            out_channels=2 * 3,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.association_vector_mag_pred = nn.Conv2d(
            in_channels=int(256*width),
            out_channels=1 * 3,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        
#         self.association_vector_unitdxy_mag_pred = nn.Conv2d(
#             in_channels=int(256*4),
#             out_channels=3 * 3,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#         )

        self.obj_pred = nn.Conv2d(
            in_channels=int(256*width),
            out_channels=1 * 3,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        
        
        
        self.num_classes = num_classes

    def forward(self, x):
        x = self.stem(x)
        cls_x = x
        reg_x = x
        asso_vec_x = x
        contact_state_x = x

        cls_feat = self.cls_conv(cls_x)
        asso_vec_feat = self.asso_vec_conv(asso_vec_x)
        contact_state_feat = self.contact_state_conv(contact_state_x)
        reg_feat = self.reg_conv(reg_x)
        
        cls_output = self.cls_pred(cls_feat)

        #handside_out = self.handside_pred(cls_feat)
        asso_vec_unitdxy_output = self.association_vector_unitdxy_pred(asso_vec_feat)
        asso_vec_mag_output = self.association_vector_mag_pred(asso_vec_feat)
        #asso_vec_unitdxy_mag_output = self.association_vector_unitdxy_mag_pred(torch.cat([asso_vec_feat, cls_feat, contact_state_feat, reg_feat], 1))
        contact_state_output = self.contact_state_pred(contact_state_feat)
        reg_output = self.reg_pred(reg_feat)
        obj_output = self.obj_pred(reg_feat)

        #x = torch.cat([reg_output, obj_output, cls_output, contact_state_output, asso_vec_unitdxy_output, asso_vec_mag_output, handside_out], 1)
        x = torch.cat([obj_output,reg_output, cls_output, contact_state_output, asso_vec_unitdxy_output, asso_vec_mag_output], 1)
        #x = torch.cat([obj_output,reg_output, cls_output, contact_state_output, asso_vec_unitdxy_mag_output], 1)

        return (
            #self.pred(x)
            x.reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    num_classes = 7
    # IMAGE_SIZE = 416
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes).cuda().eval()
    # model = YOLOv3(num_classes=num_classes).eval()
    # x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    # out = model(x)
    # assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    # assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    # assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    # print("Success!")

    x = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE)).cuda()
    torch.cuda.synchronize()
    sum_time = 0
    total_runs = 100
    warm_up = 50
    times = []
    for i in range(total_runs):
        torch.cuda.synchronize()
        start = perf_counter()
        out = model(x)
        torch.cuda.synchronize()
        end = perf_counter()
        time = int((end - start) * 1000)
        if i > warm_up:  # give torch some time to warm up
            sum_time += time
        # times.append(time)
        # print("Elapsed time: ", time, " [ms]")

    print("Avg elapsed time: ", int(sum_time / (total_runs-warm_up)), " [ms]")