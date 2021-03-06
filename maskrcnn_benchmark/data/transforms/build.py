# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.0  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    if is_train:
        aug = [
                  T.Resize(min_size, max_size),
                  T.RandomHorizontalFlip(flip_prob),
                  T.ToTensor(),
                  normalize_transform,
              ]
        if cfg.INPUT.USE_COLOR_JITTER:
            if cfg.INPUT.COLOR_JITTER_TYPE == 'imagenet':
                aug.append(T.ColorJitter())
            elif cfg.INPUT.COLOR_JITTER_TYPE == 'normal':
                def find_totensor(transform):
                    for idx, t in enumerate(transform):
                        if isinstance(t, T.ToTensor):
                            return idx
                    raise ValueError("not found totensor transform index")

                totensor_index = find_totensor(aug)
                aug.insert(totensor_index, T.ColorJitterOfficial(hue=0.05))
            else:
                raise KeyError("invalid color jitter type:{}".format(cfg.INPUT.COLOR_JITTER_TYPE))
        if cfg.INPUT.USE_EXPAND_BBOX:
            aug.append(T.ExpandBbox(cfg.INPUT.EXPAND_BBOX_FACTOR))
    else:
        aug = [
            T.Resize(min_size, max_size),  # for scale invariant
            T.ToTensor(),
            normalize_transform,
        ]
    transform = T.Compose(aug)
    return transform
