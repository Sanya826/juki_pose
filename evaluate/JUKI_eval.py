import os
import time
from collections import OrderedDict
import sys
sys.path.append('.')
import cv2
import numpy as np
import argparse
import json
import pandas as pd
from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

import pdb

import torch
import unittest
import torch

from evaluate.cocoEval import COCOeval
from lib.utils.paf_to_pose import paf_to_pose
from lib.network.rtpose_vgg import get_model, use_vgg
# from lib.network.latest_rtpose import get_model
from torch import load
from lib.datasets.preprocessing import (inception_preprocess,
                                              rtpose_preprocess,
                                              ssd_preprocess, vgg_preprocess)
from lib.network import im_transform                                              
from lib.config import cfg, update_config
from lib.utils.common import draw_results, Juki

keypoints_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='../ckpts/openpose.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)

def eval_coco(outputs, annFile, imgIds):
    """Evaluate images on Coco test set
    :param outputs: list of dictionaries, the models' processed outputs
    :param dataDir: string, path to the MSCOCO data directory
    :param imgIds: list, all the image ids in the validation set
    :returns : float, the mAP score
    """
    with open('results.json', 'w') as f:
        json.dump(outputs, f)  
    if len(outputs)==0:
        return 0.0
    cocoGt = COCO(annFile)  # load annotations
    cocoDt = cocoGt.loadRes('results.json')  # load model outputs

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    os.remove('results.json')
    # return Average Precision
    return cocoEval.stats[0]

def get_outputs(img, model, preprocess):
    """Computes the averaged heatmap and paf for the given image
    :param multiplier:
    :param origImg: numpy array, the image being processed
    :param model: pytorch model
    :returns: numpy arrays, the averaged paf and heatmap
    """
    inp_size = cfg.DATASET.IMAGE_SIZE

    # padding
    im_croped, im_scale, real_shape = im_transform.crop_with_factor(
        img, inp_size, factor=cfg.MODEL.DOWNSAMPLE, is_ceil=True)

    if preprocess == 'rtpose':
        im_data = rtpose_preprocess(im_croped)

    elif preprocess == 'vgg':
        im_data = vgg_preprocess(im_croped)

    elif preprocess == 'inception':
        im_data = inception_preprocess(im_croped)

    elif preprocess == 'ssd':
        im_data = ssd_preprocess(im_croped)

    batch_images= np.expand_dims(im_data, 0)

    # several scales as a batch
    batch_var = torch.from_numpy(batch_images).cuda().float()
    predicted_outputs, _ = model(batch_var)
    output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
    heatmap = output2.cpu().data.numpy().transpose(0, 2, 3, 1)[0]
    paf = output1.cpu().data.numpy().transpose(0, 2, 3, 1)[0]

    return paf, heatmap, im_scale

def append_result(img_id, results, upsample_keypoints, outputs):
    for result in results:
        one_result = {
            "image_id":0,
            "category_id":2,
            "keypoints":[],
            "score":0
        }
        one_result["image_id"]=img_id
        keypoints = np.zeros((12,3))
        
        all_scores = []
        for i in range(cfg.MODEL.NUM_KEYPOINTS):
            if i not in result.body_parts.keys():
                keypoints[i, 0] = 0
                keypoints[i, 1] = 0
                keypoints[i, 2] = 0
            else:
                body_part = result.body_parts[i]
                center = (body_part.x * upsample_keypoints[1] + 0.5, body_part.y * upsample_keypoints[0] + 0.5)           
                keypoints[i, 0] = center[0]
                keypoints[i, 1] = center[1]
                keypoints[i, 2] = 2

        keypoints = keypoints[keypoints_order,:]
        # pdb.set_trace()
        one_result["score"] = 1.
        one_result["keypoints"] = list(keypoints.reshape(36))

        outputs.append(one_result)

#評価の実行
def run_eval(image_dir, anno_file, vis_dir, model, preprocess):
    #アノテーションファイルの読み込み
    coco = COCO(anno_file)
    #カテゴリのidを取得
    cat_ids = coco.getCatIds(catIds=[2])
    #画像のidを取得
    img_ids = coco.getImgIds(catIds=cat_ids)
    print("Total number of validation images {}".format(len(img_ids)))
    outputs = []
    print("Processing Images in validation set")
    # 全画像で推論を開始
    for i in range(len(img_ids)):
        # 10ステップごとに出力
        if i % 10 == 0 and i != 0:
            print("Processed {} images".format(i))
        # 画像の情報を取得
        img = coco.loadImgs(img_ids[i])[0]
        # 得た情報から画像の名前を取得
        file_name = img['file_name']
        # 画像のパスを生成
        file_path = os.path.join(image_dir, file_name)
        # 画像の読み込み
        oriImg = cv2.imread(file_path)
        # 画像の幅・高さの小さいほうの値を取得
        shape_dst = np.min(oriImg.shape[0:2])

        # 推論結果を取得
        paf, heatmap, scale_img = get_outputs(oriImg, model,  preprocess)

        # 書き込むためのデータを取得する
        result = paf_to_pose(heatmap, paf, cfg)
        # pdb.set_trace()
        # 画像に結果を出力    
        out = draw_results(oriImg, result)

        # 出力用ファイルのパスを取得  
        vis_path = os.path.join(vis_dir, file_name)
        # 出力画像を保存
        cv2.imwrite(vis_path, out)
        # subset indicated how many peoples foun in this image.
        upsample_keypoints = (heatmap.shape[0]*cfg.MODEL.DOWNSAMPLE/scale_img, heatmap.shape[1]*cfg.MODEL.DOWNSAMPLE/scale_img)
        append_result(img_ids[i], result, upsample_keypoints, outputs)

    # 得た結果を使って評価値を計算
    return eval_coco(outputs=outputs, annFile=anno_file, imgIds=img_ids)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name',
                        default='./experiments/vgg19_368x368_sgd.yaml', type=str)
    parser.add_argument('--weight', type=str,
                        default='./network/weight/best_pose.pth')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # update config file
    update_config(cfg, args)

    if not os.path.exists("./data/coco/images/vis_JUKI_val"):
        os.makedirs("./data/coco/images/vis_JUKI_val")

    #Notice, if you using the 
    with torch.autograd.no_grad():
        # this path is with respect to the root of the project
        weight_name = args.weight
        state_dict = torch.load(weight_name)
            
        model = get_model(trunk='vgg19')
        #model = openpose = OpenPose_Model(l2_stages=4, l1_stages=2, paf_out_channels=38, heat_out_channels=19)
        #model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.float()
        model = model.cuda()
        
        # The choice of image preprocessing include: 'rtpose', 'inception', 'vgg' and 'ssd'.
        # If you use the converted model from caffe, it is 'rtpose' preprocess, the model trained in 
        # this repo used 'vgg' preprocess
        result = run_eval(image_dir= './data/coco/images/JUKI_val', anno_file = './data/coco/annotations/JUKI_val.json', vis_dir = './data/coco/images/vis_JUKI_val', model=model, preprocess='vgg')
    print(result)