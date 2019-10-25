import numpy as np
import cv2
import torch


class NaiveEdgeExtractor(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, images, proposals):
        imgs_edge = self._get_edge(images)
        edges_proposals = []
        for edge, proposals_per_img in zip(imgs_edge, proposals):
            edges_per_img = []
            proposals_per_img = proposals_per_img.convert('xyxy')
            for proposal in proposals_per_img:
                x1, y1, x2, y2 = proposal
                edge_per_proposal = edge[x1:x2, y1:y2]
                edges_per_img.append(edge_per_proposal)
            edges_proposals.append(edges_per_img)

        return edges_proposals

    def _get_edge(self, img):
        raise NotImplementedError
