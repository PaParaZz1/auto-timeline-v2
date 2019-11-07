import torch
import torch.nn as nn
from matcher import ArchorMatcherGCN


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
        self.sample_num = cfg.MODEL.GCN_MASK_HEAD.POLYGON_SAMPLE_NUM
        self.matcher = ArchorMatcherGCN(cfg)

    def forward(self, features, proposals, targets=None, mode='train'):
        assert(mode in ['train', 'inference'])
        if mode == 'train':
            # archor match
            matched_features, matched_targets = self.matcher(features, proposals, targets)
            # init vertex
            coordinates, neighbours_matrix = self._init_graph(matched_features)
            # sample train target
            target_vertexs = self._sample_target(targets)
            # GCN iteration
            vertexs = self._get_vertex_feature(matched_features, coordinates)
            offsets = self._forward(vertexs, neighbours_matrix, self.connected_num)
            coordinates += offsets
            # loss calculation
            raise NotImplementedError
        elif mode == 'inference':
            # init vertex
            coordinates, neighbours_matrix = self._init_graph(features)
            # GCN iteration
            for i in range(self.inference_iter):
                vertexs = self._get_vertex_feature(features, coordinates)
                offsets = self._forward(vertexs, neighbours_matrix, self.connected_num)
                coordinates += offsets
            # post process
            raise NotImplementedError
    
    def _forward(self, vertexs, neighbours_matrix, connected_num):
        raise NotImplementedError

    def _sample_target(self, targets):
        polygons = [t['masks'].polygons for t in targets]  # B images
        sample_result = []
        for idx, p in enumerate(polygons):
            num = p.polygons.shape[0] // 2
            if num == self.sample_num:
                sample_result.append(p)
            elif num > self.sample_num:
                index = random.sample([x for x in range(num)], self.sample_num)
                sample_polygons = torch.index_select(p.polygons.view(-1, 2), index, dim=0).view(-1)
                p.polygons = sample_polygons
                sample_result.append(p)
            else:
                # TODO optimize
                vacancy = self.sample_num - num
                index = random.sample([x for x in range(1, num)], vacancy)
                sample_polygons = []
                index_count = 0
                old_polygons = p.polygons.view(-1, 2)
                for idx in range(old_polygons.shape[0]):
                    sample_polygons.append(old_polygons[idx])
                    if index_count < len(index) and idx == index[index_count]:
                        sample_polygons.append((old_polygons[idx-1] + old_polygons[idx])//2)
                        index_count += 1
                p.polygons = torch.stack(sample_polygons, dim=0).view(-1)
                sample_result.append(p)
        return sample_result

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
    output = module._get_neighbours_sum(x, m, 4)
    print(output.shape)
    print(m)
    print(x[6:6+4].sum(0))
    print(output[6])


if __name__ == "__main__":
    test_get_neighbours_sum()