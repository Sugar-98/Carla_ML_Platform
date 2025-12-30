"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
from tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)




class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)#メモ：
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x) #B*N,512(upの出力チャネル),reduction_4のH,reduction_4のW
        # Depth
        x = self.depthnet(x) #B*N,self.D + self.C,reduction_4のH,reduction_4のW

        depth = self.get_depth_dist(x[:, :self.D])#深度ピン毎の確率値にする
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)#深度の確率を特徴量にかける

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))#メモ：EfficientNet の最初の層 (conv_stem + batchnorm + swish)を適用
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):#メモ：EfficientNetのブロックを順番に適用
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:#メモ：層のドロップ率が、層が深くなるほど高くなるようにする。Stochastic Depth
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):#メモ：解像度が下がったら保存する
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x #メモ：最後の出力も記録する。
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])# メモ：reduction_5の解像度を上げて、reduction_4と結合する。
        return x #B*N,512(upの出力チャネル),reduction_4のH,reduction_4のW

    def forward(self, x):#入力：B*N, 3, imH, imW
        depth, x = self.get_depth_feat(x)

        return x#出力 B*N,camC,D,imH//self.downsample, imW//self.downsample

class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, config):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = config.grid_conf
        self.data_aug_conf = config.DataLoader_config.data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64 #メモ：カメラ特徴量
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape #メモ：深度ピン数
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=config.outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)#メモ：ds=[4,5,6,7,...,45] 深度ピン
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)#メモ：画像座標系横位置
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)#メモ：画像座標系縦位置

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)#frustum型の立体の各点に(u,v,ds)のベクトルを持つベクトル場

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape #メモ：バッチ，カメラ数, (x,y,z)

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)#メモ：frustumの全u,v,dsから、post_trans分オフセットする D,H,W,3 - B,N,1,1,1,3 = B,N,D,H,W,3
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        #メモ：frustumの全u,v,dsをpost_rots分回転させる B,N,1,1,1,3,3*B,N,D,H,W,3,1 = B,N,D,H,W,3,1

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        #メモ：車両座標系のX,Yを導出する準備。(B,N,D,H,W,3(u,v,d))→(B,N,D,H,W,3(u*d,v*d,d))。x=d*u/f, y=d*v/fのため。
        combine = rots.matmul(torch.inverse(intrins))#メモ：カメラ→egoなので逆行列にする。B,N,3,3*B,N,3,3 = B,N,3,3
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)#メモ：B,N,1,1,1,3,3*B,N,D,H,W,3,1=B,N,D,H,W,3,1 → B,N,D,H,W,3
        points += trans.view(B, N, 1, 1, 1, 3)#メモ：B,N,D,H,W,3 + B,N,1,1,1,3 = B,N,D,H,W,3

        return points #メモ：B,N,D,imH/downsample,imW/downsample,3 N:カメラ数，D:深度ピン数 , 3:XYZ

    def get_cam_feats(self, x):#メモ：入力画像から深度確率Dと、特徴Cを生成
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x) #メモ：Class CamEncodeを使用して深度と特徴量を獲得
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x #B,N,D,imH//self.downsample, imW//self.downsample,C

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)#メモ：ピクセル*深度ピンの全サンプル点をフラットにする。

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        #メモ：(ego座標上の点 - (左下のボクセル中心 - ボクセルサイズ/2))/ボクセルサイズ=ボクセル範囲左下を(0,0,0)とする点/ボクセルサイズ=ボクセルインデックス(i,j,k)
        geom_feats = geom_feats.view(Nprime, 3)#メモ：ピクセル*深度ピンの全サンプル点をフラットにする。
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)#メモ：各点にバッチ番号を付与。(x_idx, y_idx, z_idx, batch_idx)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])#メモ：bound外のインデックスを除外
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]#メモ：同じバッチ内でボクセルごとに番号を割り振る。同じボクセルだと同じ番号になる。
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]#メモ：ボクセル順にデータを並べ替え

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)#メモ：同じボクセル同士で特徴量を累積和にする
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)#メモ：cumsum_trickのキャッシュを残して高速化

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device) #メモ：空のボクセルグリッド B*C*Z方向ボクセル数*X方向...*Y方向...
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x
        #メモ：geom_featsで定めたインデックスに特徴量を格納する。
        #メモ：final[batch_idx, :, z_idx, x_idx, y_idx] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)#メモ：Z軸方向を特徴量に潰す(B,C,Z,X,Y) → (B,C×Z,X,Y)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)#メモ：深度と特徴量の獲得。

        x = self.voxel_pooling(geom, x)
        #メモ：ピクセルインデックス→ボクセルインデックスに変換。特徴量を累積和にし、Z軸方向を特徴量に潰す。
        #メモ：(B,N,D,H,W,C),(B,N,D,H,W,3) → (B,C×Z,X,Y)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        #メモ： x:画像、rots:カメラの回転行列、trans:カメラの自車座標系における位置、
        #メモ：intrins:カメラ特性行列[fx,0,cx;0,fy,cy;0,0,1]、post_rots:augの回転量、post_trans:augのオフセット量
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)#メモ：BEV特徴量の取得
        x = self.bevencode(x)#メモ：特徴量からBEV画像生成
        return x #メモ：出力はBEV画像 B,outC,dH,dW


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
