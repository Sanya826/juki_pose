import os
import re
import sys
sys.path.append('.')
import cv2
import math
import glob
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.JUKI_eval import get_outputs
from lib.utils.common import draw_results, Juki
from lib.utils.paf_to_pose import paf_to_pose
from lib.config import cfg, update_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name',
                        default='./experiments/vgg19_368x368_sgd.yaml', type=str)
    parser.add_argument('--weight', type=str,
                        default='best_pose.pth')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # update config file
    update_config(cfg, args)

    if not os.path.exists("./results"):
        os.makedirs("./results")

    WEIGHT_PATH = os.path.join('./network/weight', args.weight)
    DATA_DIR = './test/'
    model = get_model('vgg19')     
    model.load_state_dict(torch.load(WEIGHT_PATH))
    model = torch.nn.DataParallel(model).cuda()
    model.float()
    model.eval()

    

    img_list = glob.glob(DATA_DIR + "*.png")
    print("Predict %d images" %len(img_list))
    count = 0
    for test_img in img_list:
        oriImg = cv2.imread(test_img) # B,G,R order
        shape_dst = np.min(oriImg.shape[0:2])

        # Get results of original image

        with torch.no_grad():
            paf, heatmap, im_scale = get_outputs(oriImg, model,  'vgg')
                
        print(im_scale)
        results = paf_to_pose(heatmap, paf, cfg)
                
        out = draw_results(oriImg, results)
        cv2.imwrite('./results/result%d.png' %count,out)
        count += 1   

