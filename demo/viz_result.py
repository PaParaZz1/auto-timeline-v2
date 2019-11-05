import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('agg')  # open it when you run this script in ssh server
import matplotlib.pyplot as plt 
import matplotlib.pylab as pylab
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets import coco
from predictor import COCODemo, draw_on_image

plt.rcParams['figure.dpi']= 250
# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12


def imsave(img, path):
    #plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(path) 


def get_mask_array(target):
    polygons = target.get_field('masks').polygons
    mask_list = [x.convert_to_binarymask() for x in polygons]
    new_list = []
    for item in mask_list:
        if item is not None:
            new_list.append(item)
    return torch.stack(new_list, dim=0).unsqueeze(1)
    

def viz(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(["TEST.IMS_PER_BATCH", "1", "MODEL.WEIGHT", args.weight_file])
    coco_demo = COCODemo(
        cfg,
        min_image_size=cfg.INPUT.MIN_SIZE_TEST,
        confidence_threshold=float(args.confidence_threshold),
    )
    VAL_DATA_DIR = '../datasets/coco/timeline/val'
    ANN_FILE = '../datasets/coco/timeline/val/coco_format.json'
    OUTPUT_DIR = '../output/' + args.config_file.split('/')[-1].split('.')[0] + '-' + args.weight_file.split('/')[-1].split('.')[0]
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    gtDataset = coco.COCODataset(ANN_FILE, VAL_DATA_DIR, True)

    for image, target, idx in gtDataset:
        image = np.array(image)
    
        top_prediction = coco_demo.compute_prediction(image)
        top_prediction = coco_demo.select_top_predictions(top_prediction)
    
        masked_image = draw_on_image(image, top_prediction)
        target.add_field('mask', get_mask_array(target))

        gt_image = draw_on_image(image, target)
        cat_img = np.concatenate([image, masked_image, gt_image], axis=1)
        imsave(cat_img, os.path.join(OUTPUT_DIR, '{}.png'.format(idx)))
        print('finish {}'.format(idx))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--weight_file')
    parser.add_argument('--confidence_threshold')
    args = parser.parse_args()

    viz(args)