import torch
import torch.nn as nn

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
        

class SelfCompleteNet4(nn.Module):  # 5raw1of
    def __init__(self, features_root=32, tot_raw_num=5, tot_of_num=1, border_mode='predict', rawRange=None, useFlow=True, padding=True):
        super(SelfCompleteNet4, self).__init__()

        assert tot_of_num <= tot_raw_num
        if border_mode == 'predict':
            self.raw_center_idx = tot_raw_num - 1
            self.of_center_idx = tot_of_num - 1
        else:
            self.raw_center_idx = (tot_raw_num - 1) // 2
            self.of_center_idx = (tot_of_num - 1) // 2
        if rawRange is None:
            self.rawRange = range(tot_raw_num)
        else:
            if rawRange < 0:
                rawRange += tot_raw_num
            assert rawRange < tot_raw_num
            self.rawRange = range(rawRange, rawRange+1)
        self.raw_channel_num = 3  # RGB channel number.
        self.of_channel_num = 2  # optical flow channel number.
        self.tot_of_num = tot_of_num
        self.tot_raw_num = tot_raw_num
        self.raw_of_offset = self.raw_center_idx - self.of_center_idx
        self.useFlow = useFlow
        self.padding = padding
        assert self.raw_of_offset >= 0

        if self.padding:
            in_channels = self.raw_channel_num * tot_raw_num
        else:
            in_channels = self.raw_channel_num * (tot_raw_num - 1)

        raw_out_channels = self.raw_channel_num
        of_out_channels = self.of_channel_num

        # Different types of incomplete video events, each corresponding to a separate UNet.
        # Raw pixel completion.
        self.inc0 = inconv(in_channels, features_root)
        self.down01 = down(features_root, features_root * 2)
        self.down02 = down(features_root * 2, features_root * 4)
        self.down03 = down(features_root * 4, features_root * 8)

        self.inc1 = inconv(in_channels, features_root)
        self.down11 = down(features_root, features_root * 2)
        self.down12 = down(features_root * 2, features_root * 4)
        self.down13 = down(features_root * 4, features_root * 8)

        self.inc2 = inconv(in_channels, features_root)
        self.down21 = down(features_root, features_root * 2)
        self.down22 = down(features_root * 2, features_root * 4)
        self.down23 = down(features_root * 4, features_root * 8)

        self.inc3 = inconv(in_channels, features_root)
        self.down31 = down(features_root, features_root * 2)
        self.down32 = down(features_root * 2, features_root * 4)
        self.down33 = down(features_root * 4, features_root * 8)

        self.inc4 = inconv(in_channels, features_root)
        self.down41 = down(features_root, features_root * 2)
        self.down42 = down(features_root * 2, features_root * 4)
        self.down43 = down(features_root * 4, features_root * 8)

        self.up01 = up(features_root * 8, features_root * 4)
        self.up02 = up(features_root * 4, features_root * 2)
        self.up03 = up(features_root * 2, features_root)
        self.outc0 = outconv(features_root, raw_out_channels)

        self.up11 = up(features_root * 8, features_root * 4)
        self.up12 = up(features_root * 4, features_root * 2)
        self.up13 = up(features_root * 2, features_root)
        self.outc1 = outconv(features_root, raw_out_channels)

        self.up21 = up(features_root * 8, features_root * 4)
        self.up22 = up(features_root * 4, features_root * 2)
        self.up23 = up(features_root * 2, features_root)
        self.outc2 = outconv(features_root, raw_out_channels)

        self.up31 = up(features_root * 8, features_root * 4)
        self.up32 = up(features_root * 4, features_root * 2)
        self.up33 = up(features_root * 2, features_root)
        self.outc3 = outconv(features_root, raw_out_channels)

        self.up41 = up(features_root * 8, features_root * 4)
        self.up42 = up(features_root * 4, features_root * 2)
        self.up43 = up(features_root * 2, features_root)
        self.outc4 = outconv(features_root, raw_out_channels)

        # Optical flow completion.
        if useFlow:
            self.inc_of = inconv(in_channels, features_root)
            self.down_of1 = down(features_root, features_root * 2)
            self.down_of2 = down(features_root * 2, features_root * 4)
            self.down_of3 = down(features_root * 4, features_root * 8)

            self.up_of1 = up(features_root * 8, features_root * 4)
            self.up_of2 = up(features_root * 4, features_root * 2)
            self.up_of3 = up(features_root * 2, features_root)
            self.outc_of = outconv(features_root, of_out_channels)

    def forward(self, x, x_of):
        # Use incomplete inputs to yield complete inputs.
        all_raw_outputs = []
        all_raw_targets = []
        all_of_outputs = []
        all_of_targets = []
        for raw_i in self.rawRange:
            if self.padding:
                incomplete_x = x.clone()
                incomplete_x[:, raw_i * self.raw_channel_num:(raw_i + 1) * self.raw_channel_num, :, :] = 0
            else:
                incomplete_x = torch.cat([x[:, :raw_i * self.raw_channel_num, :, :], x[:, (raw_i+1) * self.raw_channel_num: , :, :]], dim=1)
            all_raw_targets.append(x[:, raw_i * self.raw_channel_num:(raw_i + 1) * self.raw_channel_num, :, :])

            # Complete video events (raw pixel).
            if raw_i == 0:
                x1 = self.inc0(incomplete_x)
                x2 = self.down01(x1)
                x3 = self.down02(x2)
                x4 = self.down03(x3)

                raw = self.up01(x4, x3)
                raw = self.up02(raw, x2)
                raw = self.up03(raw, x1)
                raw = self.outc0(raw)
                all_raw_outputs.append(raw)
            elif raw_i == 1:
                x1 = self.inc1(incomplete_x)
                x2 = self.down11(x1)
                x3 = self.down12(x2)
                x4 = self.down13(x3)

                raw = self.up11(x4, x3)
                raw = self.up12(raw, x2)
                raw = self.up13(raw, x1)
                raw = self.outc1(raw)
                all_raw_outputs.append(raw)
            elif raw_i == 2:
                x1 = self.inc2(incomplete_x)
                x2 = self.down21(x1)
                x3 = self.down22(x2)
                x4 = self.down23(x3)

                raw = self.up21(x4, x3)
                raw = self.up22(raw, x2)
                raw = self.up23(raw, x1)
                raw = self.outc2(raw)
                all_raw_outputs.append(raw)
            elif raw_i == 3:
                x1 = self.inc3(incomplete_x)
                x2 = self.down31(x1)
                x3 = self.down32(x2)
                x4 = self.down33(x3)

                raw = self.up31(x4, x3)
                raw = self.up32(raw, x2)
                raw = self.up33(raw, x1)
                raw = self.outc3(raw)
                all_raw_outputs.append(raw)
            elif raw_i == 4:
                x1 = self.inc4(incomplete_x)
                x2 = self.down41(x1)
                x3 = self.down42(x2)
                x4 = self.down43(x3)

                raw = self.up41(x4, x3)
                raw = self.up42(raw, x2)
                raw = self.up43(raw, x1)
                raw = self.outc4(raw)
                all_raw_outputs.append(raw)
            else:
                print('out of range！')
                raise NotImplementedError

            # Complete video events (optical flow).
            of_i = raw_i - self.raw_of_offset
            if self.useFlow and 0 <= of_i < self.tot_of_num:
                ofx1 = self.inc_of(incomplete_x)
                ofx2 = self.down_of1(ofx1)
                ofx3 = self.down_of2(ofx2)
                ofx4 = self.down_of3(ofx3)

                of = self.up_of1(ofx4, ofx3)
                of = self.up_of2(of, ofx2)
                of = self.up_of3(of, ofx1)
                of = self.outc_of(of)
                all_of_outputs.append(of)
                all_of_targets.append(x_of[:, of_i * self.of_channel_num:(of_i + 1) * self.of_channel_num, :, :])

        all_raw_outputs = torch.cat(all_raw_outputs, dim=1)
        all_raw_targets = torch.cat(all_raw_targets, dim=1)
        if len(all_of_outputs) > 0:
            all_of_outputs = torch.cat(all_of_outputs, dim=1)
            all_of_targets = torch.cat(all_of_targets, dim=1)

        return all_of_outputs, all_raw_outputs, all_of_targets, all_raw_targets


class SelfCompleteNetFull(nn.Module):  # 5raw5of
    def __init__(self, features_root=32, tot_raw_num=5, tot_of_num=5, border_mode='predict', rawRange=None, useFlow=True, padding=True):
        super(SelfCompleteNetFull, self).__init__()
        assert tot_of_num <= tot_raw_num
        if border_mode == 'predict' or border_mode == 'elasticPredict':
            self.raw_center_idx = tot_raw_num - 1
            self.of_center_idx = tot_of_num - 1
        else:
            self.raw_center_idx = (tot_raw_num - 1) // 2
            self.of_center_idx = (tot_of_num - 1) // 2
        if rawRange is None:
            self.rawRange = range(tot_raw_num)
        else:
            if rawRange < 0:
                rawRange += tot_raw_num
            assert rawRange < tot_raw_num
            self.rawRange = range(rawRange, rawRange+1)
        self.raw_channel_num = 3  # RGB channel number.
        self.of_channel_num = 2  # optical flow channel number.
        self.tot_of_num = tot_of_num
        self.tot_raw_num = tot_raw_num
        
        self.raw_of_offset = self.raw_center_idx - self.of_center_idx
        
        self.useFlow = useFlow
        self.padding = padding
        assert self.raw_of_offset >= 0

        if self.padding:
            in_channels = self.raw_channel_num * tot_raw_num
        else:
            in_channels = self.raw_channel_num * (tot_raw_num - 1)

        raw_out_channels = self.raw_channel_num
        of_out_channels = self.of_channel_num

        # Different types of incomplete video events, each corresponding to a separate UNet.
        # Raw pixel completion.
        self.inc0 = inconv(in_channels, features_root)
        self.down01 = down(features_root, features_root * 2)
        self.down02 = down(features_root * 2, features_root * 4)
        self.down03 = down(features_root * 4, features_root * 8)

        self.inc1 = inconv(in_channels, features_root)
        self.down11 = down(features_root, features_root * 2)
        self.down12 = down(features_root * 2, features_root * 4)
        self.down13 = down(features_root * 4, features_root * 8)

        self.inc2 = inconv(in_channels, features_root)
        self.down21 = down(features_root, features_root * 2)
        self.down22 = down(features_root * 2, features_root * 4)
        self.down23 = down(features_root * 4, features_root * 8)

        self.inc3 = inconv(in_channels, features_root)
        self.down31 = down(features_root, features_root * 2)
        self.down32 = down(features_root * 2, features_root * 4)
        self.down33 = down(features_root * 4, features_root * 8)

        self.inc4 = inconv(in_channels, features_root)
        self.down41 = down(features_root, features_root * 2)
        self.down42 = down(features_root * 2, features_root * 4)
        self.down43 = down(features_root * 4, features_root * 8)

        self.up01 = up(features_root * 8, features_root * 4)
        self.up02 = up(features_root * 4, features_root * 2)
        self.up03 = up(features_root * 2, features_root)
        self.outc0 = outconv(features_root, raw_out_channels)

        self.up11 = up(features_root * 8, features_root * 4)
        self.up12 = up(features_root * 4, features_root * 2)
        self.up13 = up(features_root * 2, features_root)
        self.outc1 = outconv(features_root, raw_out_channels)

        self.up21 = up(features_root * 8, features_root * 4)
        self.up22 = up(features_root * 4, features_root * 2)
        self.up23 = up(features_root * 2, features_root)
        self.outc2 = outconv(features_root, raw_out_channels)

        self.up31 = up(features_root * 8, features_root * 4)
        self.up32 = up(features_root * 4, features_root * 2)
        self.up33 = up(features_root * 2, features_root)
        self.outc3 = outconv(features_root, raw_out_channels)

        self.up41 = up(features_root * 8, features_root * 4)
        self.up42 = up(features_root * 4, features_root * 2)
        self.up43 = up(features_root * 2, features_root)
        self.outc4 = outconv(features_root, raw_out_channels)

        # Optical flow completion.
        if useFlow:
            self.inc_of0 = inconv(in_channels, features_root)
            self.down_of01 = down(features_root, features_root * 2)
            self.down_of02 = down(features_root * 2, features_root * 4)
            self.down_of03 = down(features_root * 4, features_root * 8)

            self.inc_of1 = inconv(in_channels, features_root)
            self.down_of11 = down(features_root, features_root * 2)
            self.down_of12 = down(features_root * 2, features_root * 4)
            self.down_of13 = down(features_root * 4, features_root * 8)

            self.inc_of2 = inconv(in_channels, features_root)
            self.down_of21 = down(features_root, features_root * 2)
            self.down_of22 = down(features_root * 2, features_root * 4)
            self.down_of23 = down(features_root * 4, features_root * 8)

            self.inc_of3 = inconv(in_channels, features_root)
            self.down_of31 = down(features_root, features_root * 2)
            self.down_of32 = down(features_root * 2, features_root * 4)
            self.down_of33 = down(features_root * 4, features_root * 8)

            self.inc_of4 = inconv(in_channels, features_root)
            self.down_of41 = down(features_root, features_root * 2)
            self.down_of42 = down(features_root * 2, features_root * 4)
            self.down_of43 = down(features_root * 4, features_root * 8)

            self.up_of01 = up(features_root * 8, features_root * 4)
            self.up_of02 = up(features_root * 4, features_root * 2)
            self.up_of03 = up(features_root * 2, features_root)
            self.outc_of0 = outconv(features_root, of_out_channels)

            self.up_of11 = up(features_root * 8, features_root * 4)
            self.up_of12 = up(features_root * 4, features_root * 2)
            self.up_of13 = up(features_root * 2, features_root)
            self.outc_of1 = outconv(features_root, of_out_channels)

            self.up_of21 = up(features_root * 8, features_root * 4)
            self.up_of22 = up(features_root * 4, features_root * 2)
            self.up_of23 = up(features_root * 2, features_root)
            self.outc_of2 = outconv(features_root, of_out_channels)

            self.up_of31 = up(features_root * 8, features_root * 4)
            self.up_of32 = up(features_root * 4, features_root * 2)
            self.up_of33 = up(features_root * 2, features_root)
            self.outc_of3 = outconv(features_root, of_out_channels)

            self.up_of41 = up(features_root * 8, features_root * 4)
            self.up_of42 = up(features_root * 4, features_root * 2)
            self.up_of43 = up(features_root * 2, features_root)
            self.outc_of4 = outconv(features_root, of_out_channels)

    def forward(self, x, x_of):
        # Use incomplete inputs to yield complete inputs.
        all_raw_outputs = []
        all_raw_targets = []
        all_of_outputs = []
        all_of_targets = []
        for raw_i in self.rawRange:
            if self.padding:
                incomplete_x = x.clone()
                incomplete_x[:, raw_i * self.raw_channel_num:(raw_i + 1) * self.raw_channel_num, :, :] = 0
            else:
                incomplete_x = torch.cat([x[:, :raw_i * self.raw_channel_num, :, :], x[:, (raw_i+1) * self.raw_channel_num: , :, :]], dim=1)
            all_raw_targets.append(x[:, raw_i * self.raw_channel_num:(raw_i + 1) * self.raw_channel_num, :, :])

            if raw_i == 0:
                x1 = self.inc0(incomplete_x)
                x2 = self.down01(x1)
                x3 = self.down02(x2)
                x4 = self.down03(x3)

                raw = self.up01(x4, x3)
                raw = self.up02(raw, x2)
                raw = self.up03(raw, x1)
                raw = self.outc0(raw)
                all_raw_outputs.append(raw)
            elif raw_i == 1:
                x1 = self.inc1(incomplete_x)
                x2 = self.down11(x1)
                x3 = self.down12(x2)
                x4 = self.down13(x3)

                raw = self.up11(x4, x3)
                raw = self.up12(raw, x2)
                raw = self.up13(raw, x1)
                raw = self.outc1(raw)
                all_raw_outputs.append(raw)
            elif raw_i == 2:
                x1 = self.inc2(incomplete_x)
                x2 = self.down21(x1)
                x3 = self.down22(x2)
                x4 = self.down23(x3)

                raw = self.up21(x4, x3)
                raw = self.up22(raw, x2)
                raw = self.up23(raw, x1)
                raw = self.outc2(raw)
                all_raw_outputs.append(raw)
            elif raw_i == 3:
                x1 = self.inc3(incomplete_x)
                x2 = self.down31(x1)
                x3 = self.down32(x2)
                x4 = self.down33(x3)

                raw = self.up31(x4, x3)
                raw = self.up32(raw, x2)
                raw = self.up33(raw, x1)
                raw = self.outc3(raw)
                all_raw_outputs.append(raw)
            elif raw_i == 4:
                x1 = self.inc4(incomplete_x)
                x2 = self.down41(x1)
                x3 = self.down42(x2)
                x4 = self.down43(x3)

                raw = self.up41(x4, x3)
                raw = self.up42(raw, x2)
                raw = self.up43(raw, x1)
                raw = self.outc4(raw)
                all_raw_outputs.append(raw)
            else:
                print('out of range！')
                raise NotImplementedError

            of_i = raw_i - self.raw_of_offset
            if self.useFlow and 0 <= of_i < self.tot_of_num:
                if of_i == 0:
                    ofx1 = self.inc_of0(incomplete_x)
                    ofx2 = self.down_of01(ofx1)
                    ofx3 = self.down_of02(ofx2)
                    ofx4 = self.down_of03(ofx3)

                    of = self.up_of01(ofx4, ofx3)
                    of = self.up_of02(of, ofx2)
                    of = self.up_of03(of, ofx1)
                    of = self.outc_of0(of)

                    all_of_outputs.append(of)
                elif of_i == 1:
                    ofx1 = self.inc_of1(incomplete_x)
                    ofx2 = self.down_of11(ofx1)
                    ofx3 = self.down_of12(ofx2)
                    ofx4 = self.down_of13(ofx3)

                    of = self.up_of11(ofx4, ofx3)
                    of = self.up_of12(of, ofx2)
                    of = self.up_of13(of, ofx1)
                    of = self.outc_of1(of)

                    all_of_outputs.append(of)
                elif of_i == 2:
                    ofx1 = self.inc_of2(incomplete_x)
                    ofx2 = self.down_of21(ofx1)
                    ofx3 = self.down_of22(ofx2)
                    ofx4 = self.down_of23(ofx3)

                    of = self.up_of21(ofx4, ofx3)
                    of = self.up_of22(of, ofx2)
                    of = self.up_of23(of, ofx1)
                    of = self.outc_of2(of)

                    all_of_outputs.append(of)
                elif of_i == 3:
                    ofx1 = self.inc_of3(incomplete_x)
                    ofx2 = self.down_of31(ofx1)
                    ofx3 = self.down_of32(ofx2)
                    ofx4 = self.down_of33(ofx3)

                    of = self.up_of31(ofx4, ofx3)
                    of = self.up_of32(of, ofx2)
                    of = self.up_of33(of, ofx1)
                    of = self.outc_of3(of)

                    all_of_outputs.append(of)
                elif of_i == 4:
                    ofx1 = self.inc_of4(incomplete_x)
                    ofx2 = self.down_of41(ofx1)
                    ofx3 = self.down_of42(ofx2)
                    ofx4 = self.down_of43(ofx3)

                    of = self.up_of41(ofx4, ofx3)
                    of = self.up_of42(of, ofx2)
                    of = self.up_of43(of, ofx1)
                    of = self.outc_of4(of)

                    all_of_outputs.append(of)
                else:
                    print('out of optical flow range！')
                    raise NotImplementedError
                all_of_targets.append(x_of[:, of_i * self.of_channel_num:(of_i + 1) * self.of_channel_num, :, :])

        all_raw_outputs = torch.cat(all_raw_outputs, dim=1)
        all_raw_targets = torch.cat(all_raw_targets, dim=1)
        if len(all_of_outputs) > 0:
            all_of_outputs = torch.cat(all_of_outputs, dim=1)
            all_of_targets = torch.cat(all_of_targets, dim=1)

        return all_of_outputs, all_raw_outputs, all_of_targets, all_raw_targets


class SelfCompleteNet1raw1of(nn.Module):  # 1raw1of
    '''
    rawRange: Int, the idx of raw inputs to be predicted
    '''
    def __init__(self, features_root=64, tot_raw_num=5, tot_of_num=1, border_mode='predict', rawRange=None, useFlow=True, padding=True):
        super(SelfCompleteNet1raw1of, self).__init__()

        assert tot_of_num <= tot_raw_num
        if border_mode == 'predict':
            self.raw_center_idx = tot_raw_num - 1
            self.of_center_idx = tot_of_num - 1
        else:
            self.raw_center_idx = (tot_raw_num - 1) // 2
            self.of_center_idx = (tot_of_num - 1) // 2
        if rawRange is None:
            self.rawRange = range(tot_raw_num)
        else:
            if rawRange < 0:
                rawRange += tot_raw_num
            assert rawRange < tot_raw_num
            self.rawRange = range(rawRange, rawRange+1)
        self.raw_channel_num = 3  # RGB channel no.
        self.of_channel_num = 2  # optical flow channel no.
        self.tot_of_num = tot_of_num
        self.tot_raw_num = tot_raw_num
        self.raw_of_offset = self.raw_center_idx - self.of_center_idx
        
        self.useFlow = useFlow
        self.padding = padding
        assert self.raw_of_offset >= 0

        if self.padding:
            in_channels = self.raw_channel_num * tot_raw_num
        else:
            in_channels = self.raw_channel_num * (tot_raw_num - 1)

        raw_out_channels = self.raw_channel_num
        of_out_channels = self.of_channel_num

        self.inc = inconv(in_channels, features_root)
        self.down1 = down(features_root, features_root * 2)
        self.down2 = down(features_root * 2, features_root * 4)
        self.down3 = down(features_root * 4, features_root * 8)
        # 0
        self.up1 = up(features_root * 8, features_root * 4)
        self.up2 = up(features_root * 4, features_root * 2)
        self.up3 = up(features_root * 2, features_root)
        self.outc = outconv(features_root, raw_out_channels)

        if useFlow:
            self.inc_of = inconv(in_channels, features_root)
            self.down_of1 = down(features_root, features_root * 2)
            self.down_of2 = down(features_root * 2, features_root * 4)
            self.down_of3 = down(features_root * 4, features_root * 8)

            self.up_of1 = up(features_root * 8, features_root * 4)
            self.up_of2 = up(features_root * 4, features_root * 2)
            self.up_of3 = up(features_root * 2, features_root)
            self.outc_of = outconv(features_root, of_out_channels)

    def forward(self, x, x_of):
        # use incomplete inputs to yield complete inputs
        if self.padding:
            incomplete_x = x.clone()
            incomplete_x[:, (self.tot_raw_num - 1) * self.raw_channel_num: , :, :] = 0
        else:
            incomplete_x = x[:, :(self.tot_raw_num - 1) * self.raw_channel_num, :, :]
        raw_target = x[:, (self.tot_raw_num - 1) * self.raw_channel_num: , :, :]
        
        x1 = self.inc(incomplete_x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        raw = self.up1(x4, x3)
        raw = self.up2(raw, x2)
        raw = self.up3(raw, x1)
        raw_output = self.outc(raw)
        
        of_i = self.tot_raw_num - 1 - self.raw_of_offset
        if self.useFlow:
            ofx1 = self.inc_of(incomplete_x)
            ofx2 = self.down_of1(ofx1)
            ofx3 = self.down_of2(ofx2)
            ofx4 = self.down_of3(ofx3)

            of = self.up_of1(ofx4, ofx3)
            of = self.up_of2(of, ofx2)
            of = self.up_of3(of, ofx1)
            of_output = self.outc_of(of)
        
            of_target = x_of[:, of_i * self.of_channel_num:(of_i + 1) * self.of_channel_num, :, :]
        
        return of_output, raw_output, of_target, raw_target


