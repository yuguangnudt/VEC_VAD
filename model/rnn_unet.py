import torch
import torch.nn as nn
from itertools import combinations
from model.rnn import ConvLSTM, ConvGRU


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    '''
    inconv only changes the number of channels
    '''
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            double_conv(in_ch, out_ch),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        self.bilinear=bilinear
        if self.bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch//2, 1),)
        else:
            self.up =  nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class LSTM_Unet_Cnm3(nn.Module):
    '''
    rawRange: Int, the idx of raw inputs to be predicted
    m: the number of erased patches
    '''

    def __init__(self, features_root=32, tot_raw_num=5, tot_of_num=5, border_mode='predict', rawRange=None,
                 useFlow=True, padding=True, m=1):
        super(LSTM_Unet_Cnm3, self).__init__()
        assert tot_of_num == tot_raw_num
        if rawRange is None:
            self.rawRange = range(tot_raw_num)
        else:
            if rawRange < 0:
                rawRange += tot_raw_num
            assert rawRange < tot_raw_num
            self.rawRange = range(rawRange, rawRange + 1)
        self.raw_channel_num = 3  # RGB channel no.
        self.of_channel_num = 2  # optical flow channel no.
        self.tot_of_num = tot_of_num
        self.tot_raw_num = tot_raw_num

        self.useFlow = useFlow
        self.padding = padding

        raw_out_channels = self.raw_channel_num * m
        of_out_channels = self.of_channel_num * m

        self.erase_indexes = [list(c) for c in combinations(self.rawRange, m)]  # C(n, m)

        self.inconvlstm_modules = nn.ModuleList([ConvLSTM(input_dim=self.raw_channel_num,
                                                          hidden_dim=features_root,
                                                          kernel_size=(3, 3),
                                                          num_layers=1,
                                                          batch_first=True,
                                                          bias=True,
                                                          return_all_layers=False)
                                                 for i in range(len(self.erase_indexes))])

        self.down1_modules = nn.ModuleList([down(features_root, features_root * 2) for i in range(len(self.erase_indexes))])
        self.down2_modules = nn.ModuleList([down(features_root * 2, features_root * 4) for i in range(len(self.erase_indexes))])
        self.down3_modules = nn.ModuleList([down(features_root * 4, features_root * 8) for i in range(len(self.erase_indexes))])
        self.up1_modules = nn.ModuleList([up(features_root * 8, features_root * 4) for i in range(len(self.erase_indexes))])
        self.up2_modules = nn.ModuleList([up(features_root * 4, features_root * 2) for i in range(len(self.erase_indexes))])
        self.up3_modules = nn.ModuleList([up(features_root * 2, features_root) for i in range(len(self.erase_indexes))])
        self.outconv_modules = nn.ModuleList([outconv(features_root, raw_out_channels) for i in range(len(self.erase_indexes))])

        if useFlow:
            self.inconvlstm_of_modules = nn.ModuleList([ConvLSTM(input_dim=self.raw_channel_num,
                                                                 hidden_dim=features_root,
                                                                 kernel_size=(3, 3),
                                                                 num_layers=1,
                                                                 batch_first=True,
                                                                 bias=True,
                                                                 return_all_layers=False)
                                                        for i in range(len(self.erase_indexes))])

            self.down1_of_modules = nn.ModuleList([down(features_root, features_root * 2) for i in range(len(self.erase_indexes))])
            self.down2_of_modules = nn.ModuleList([down(features_root * 2, features_root * 4) for i in range(len(self.erase_indexes))])
            self.down3_of_modules = nn.ModuleList([down(features_root * 4, features_root * 8) for i in range(len(self.erase_indexes))])
            self.up1_of_modules = nn.ModuleList([up(features_root * 8, features_root * 4) for i in range(len(self.erase_indexes))])
            self.up2_of_modules = nn.ModuleList([up(features_root * 4, features_root * 2) for i in range(len(self.erase_indexes))])
            self.up3_of_modules = nn.ModuleList([up(features_root * 2, features_root) for i in range(len(self.erase_indexes))])
            self.outconv_of_modules = nn.ModuleList([outconv(features_root, of_out_channels) for i in range(len(self.erase_indexes))])

    def forward(self, x, x_of):
        # use incomplete inputs to yield complete inputs
        all_raw_outputs = []
        all_raw_targets = []
        all_of_outputs = []
        all_of_targets = []
        for idx, c in enumerate(self.erase_indexes):
            for i in c:
                all_raw_targets.append(x[:, i, :, :, :])
                all_of_targets.append(x_of[:, i, :, :, :])

            remain_idx = list(set(self.rawRange).difference(set(c)))  # difference set
            remain_idx.sort(key=self.rawRange.index)  # original order
            if self.padding:
                incomplete_x = x.clone()
                for i in c:
                    incomplete_x[:, i:(i + 1), :, :, :] = 0
            else:
                incomplete_x = []
                for i in remain_idx:
                    incomplete_x.append(x[:, i:(i+1), :, :, :])
                incomplete_x = torch.cat(incomplete_x, dim=1)

            #  raw complete
            layer_output, _ = self.inconvlstm_modules[idx](incomplete_x)

            x1 = torch.sum(layer_output[0], dim=1)
            x2 = self.down1_modules[idx](x1)
            x3 = self.down2_modules[idx](x2)
            x4 = self.down3_modules[idx](x3)

            raw = self.up1_modules[idx](x4, x3)
            raw = self.up2_modules[idx](raw, x2)
            raw = self.up3_modules[idx](raw, x1)
            raw = self.outconv_modules[idx](raw)
            all_raw_outputs.append(raw)

            # optical flow complete
            if self.useFlow:
                layer_output, _ = self.inconvlstm_of_modules[idx](incomplete_x)

                ofx1 = torch.sum(layer_output[0], dim=1)
                ofx2 = self.down1_of_modules[idx](ofx1)
                ofx3 = self.down2_of_modules[idx](ofx2)
                ofx4 = self.down3_of_modules[idx](ofx3)

                of = self.up1_of_modules[idx](ofx4, ofx3)
                of = self.up2_of_modules[idx](of, ofx2)
                of = self.up3_of_modules[idx](of, ofx1)
                of = self.outconv_of_modules[idx](of)
                all_of_outputs.append(of)

        all_raw_outputs = torch.cat(all_raw_outputs, dim=1)
        all_raw_targets = torch.cat(all_raw_targets, dim=1)
        if len(all_of_outputs) > 0:
            all_of_outputs = torch.cat(all_of_outputs, dim=1)
            all_of_targets = torch.cat(all_of_targets, dim=1)

        return all_of_outputs, all_raw_outputs, all_of_targets, all_raw_targets
