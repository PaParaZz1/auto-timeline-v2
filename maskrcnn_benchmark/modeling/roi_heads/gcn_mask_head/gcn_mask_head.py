import torch
import torch.nn as nn


class GCNResNetLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=nn.ReLU(), norm=None, use_se=False):
        super(GCNResNetLayer, self).__init__()
        self.conv_x0 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.conv_n0 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.conv_x1 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.conv_n1 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.act = activation
        if norm is not None:
            raise NotImplementedError
        if use_se:
            raise NotImplementedError

    def _get_neighbours_sum(self, n, m, connected_num):
        B, L = m.shape[:2]
        N, C, H, W = n.shape
        n_broadcast = n.view(B, L, C, H, W).unsqueeze(1)
        mask = m.reshape(B, L, L, 1, 1, 1)
        o = torch.masked_select(n_broadcast, mask) 
        o = o.view(B, L, connected_num, C, H, W)
        return o.sum(dim=2).view(N, C, H, W)

    def forward(self, x, neighbours_matrix, connected_num=4):
        m = neighbours_matrix
        residual = x
        n_set = self.conv_n0(x)
        x = self.conv_x0(x)
        x = self.act(x + self._get_neighbours_sum(n_set, m, connected_num))  

        n_set = self.conv_n1(x)
        x = self.conv_x1(x)
        x = x + self._get_neighbours_sum(n_set, m, connected_num)
        return self.act(x + residual)


class ROIGCNMaskHead(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIGCNMaskHead, self).__init__()
        self.inference_iter = cfg.MODEL.GCN_MASK_HEAD.INFERNECE_ITER
        self.connected_num = cfg.MODEL.GCN_MASK_HEAD.CONNECTED_NUM

    def forward(self, features, targets, mode):
        assert(mode in ['train', 'inference'])
        # init vertex location
        coordinates, neighbours_matrix = self._init_graph(features)
        if mode == 'train':
            vertexs = self._get_vertex_feature(features, coordinates)
            offsets = self._forward(vertexs, neighbours_matrix, self.connected_num)
            coordinates += offsets
            # loss calculation
            raise NotImplementedError
        elif mode == 'inference':
            for i in range(self.inference_iter):
                vertexs = self._get_vertex_feature(features, coordinates)
                offsets = self._forward(vertexs, neighbours_matrix, self.connected_num)
                coordinates += offsets
            # post process
            raise NotImplementedError
    
    def _forward(self, vertexs, neighbours_matrix, connected_num):
        raise NotImplementedError

    def _init_graph(self, features):
        raise NotImplementedError

    def _get_vertex_feature(self, features, coordinates):
        raise NotImplementedError


def build_gcn_mask_head(cfg):
    return ROIGCNMaskHead(cfg)


def test_get_neighbours_sum():
    module = GCNResNetLayer(3, 3, 3)
    x = torch.randn(2*6, 3, 4, 4)
    m = torch.ByteTensor([[1, 0, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]]).view(2, 1, 6).repeat(1, 6, 1)
    output = module._get_neighbours_sum(x, m)
    print(output.shape)
    print(m)
    print(x[6:6+4].sum(0))
    print(output[6])


if __name__ == "__main__":
    test_get_neighbours_sum()