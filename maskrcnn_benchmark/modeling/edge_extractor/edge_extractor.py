import numpy as np
import cv2
import torch


def getY(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return yuv[..., 0]


def extract_grad(img):
    return cv2.Laplacian(img, cv2.CV_8U)


def get_edge_mask_np(img):
    '''
        Args:
            img: np.ndarray(H, W, 3) (0-255)(uint8)
        Return:
            y_channel: np.ndarray(H, W, 1) (0 or 255)(uint8)
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img = cv2.GaussianBlur(img, (5, 5), 0)

    y_channel = getY(img)
    count_num = np.argmax(np.bincount(y_channel.reshape(-1)))  # background color mapping
    y_channel[y_channel == 255] = 254
    y_channel[np.abs(y_channel - count_num) < 5] = 255

    y_grad = extract_grad(y_channel)

    y_channel = cv2.equalizeHist(y_channel)
    y_channel = cv2.morphologyEx(y_channel, cv2.MORPH_CLOSE, kernel)

    y_channel = np.where(y_grad > 3, y_channel, 255)

    y_channel = np.where(y_channel < 200, 255, 0)  # binarize
    return np.expand_dims(y_channel, axis=2)


class NaiveEdgeExtractor(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, images, proposals):
        imgs_edge = self._get_edge(images)
        edges_proposals = []
        for edge, proposals_per_img in zip(imgs_edge, proposals):
            edges_per_img = []
            proposals_per_img = proposals_per_img.convert('xyxy')
            for proposal in proposals_per_img.bbox:
                x1, y1, x2, y2 = proposal.round().long()
                if x1 == x2 or y1 == y2:
                    x2 += 2
                    y2 += 2
                    #print('invalid proposal(equal coordinate)', proposal)
                if self._out_of_img_range(edge.shape, proposal):
                    edge_per_proposal = edge[:, x1:x2, y1:y2]
                else:
                    edge_per_proposal = torch.zeros(1, x2-x1, y2-y1).to(edge.device)
                if 0 in edge_per_proposal.shape:
                    print('invalid proposal(out of img range)', proposal)
                    print(edge.shape)
                    print(edge_per_proposal.shape)
                edges_per_img.append(edge_per_proposal)
            edges_proposals.append(edges_per_img)

        return edges_proposals

    def _out_of_img_range(self, shape, proposals):
        H, W = shape[1:]
        x1, y1, x2, y2 = proposals
        flag = ((0 <= x1 and x1 < H) and
                (0 <= x2 and x2 < H) and
                (0 <= y1 and y1 < W) and
                (0 <= y2 and y2 < W))
        return flag

    def _get_edge(self, imgs):
        device = imgs.tensors.device
        imgs_edge = []
        np_imgs = imgs.tensors.mul_(255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        for img in np_imgs:
            edges = get_edge_mask_np(img)
            imgs_edge.append(edges)
        imgs_edge = np.stack(imgs_edge, axis=0)
        imgs_edge = torch.FloatTensor(imgs_edge).to(device).permute(0, 3, 1, 2).div_(255)
        return imgs_edge