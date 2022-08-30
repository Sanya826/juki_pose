import json
import shutil
import os
from pycocotools.coco import COCO
import numpy as np

#アノテーションファイルを作成する
def make_json_file(imgs, anns, cats, prefix="train"):
    data = {"images":imgs, "categories":cats, "annotations":anns}
    with open("./data/coco/annotations/JUKI_%s.json"%prefix, 'w') as f:
        json.dump(data, f, indent=0, ensure_ascii=False)

def make_img_file(img_data, img_dir, orig_dir_name, prefix="train"):
    dir_name = orig_dir_name + "_" + prefix
    new_img_dir = os.path.join(img_dir, dir_name)
    orig_img_dir = os.path.join(img_dir, orig_dir_name)
    if not os.path.exists(new_img_dir):
        os.makedirs(new_img_dir)
    for img in img_data:
        file_path = os.path.join(orig_img_dir, img["file_name"])
        shutil.copy(file_path, new_img_dir)

def add_area(img,ann):
    if "A-1-1" in img["file_name"]:
        ann["area"] = 122000
    elif "A-1-2" in img["file_name"]:
        ann["area"] = 98800
    elif "A-2-1" in img["file_name"]:
        ann["area"] = 94000
    elif "A-2-2" in img["file_name"]:
        ann["area"] = 83500
    else:
        ann["area"] = 83000
    return ann

if __name__=="__main__":
    #JSONファイルの読み込み
    annFile = "./data/coco/annotations/JUKI.json"
    IMAGE_DIR = "./data/coco/images/"
    ORIG_DIR_NAME = "JUKI"
    coco = COCO(annFile)
    #画像データを取得
    imgs = coco.imgs
    #annotationデータを取得(画像idをもとに分ける)
    anns = coco.imgToAnns
    #categoryの取得
    cats = list(coco.cats.values())
    #画像のidを取得する
    img_ids = list(imgs.keys())
    #全体の数を取得
    all_img_num = len(img_ids)
    #全体の3割を評価データに使用する
    val_img_num = all_img_num//3
    #残りのデータを学習データにする
    train_img_num = all_img_num - val_img_num
    test_img_num = 816
    #それぞれの数を出力
    print("All image:", all_img_num, "\n")
    print("Train image:", train_img_num, "\n")
    print("Val image:", val_img_num, "\n")
    #shuffle
    shuffle_img_ids = np.random.permutation(img_ids)
    img_count = 0
    img_data = []
    ann_data = []
    #annotation file
    for ids in shuffle_img_ids:
        img_data.append(imgs[ids])
        ann_data.append(add_area(imgs[ids],anns[ids][0]))
        img_count += 1
        if img_count == val_img_num:
            print("Make Val File...")
            make_img_file(img_data, IMAGE_DIR, ORIG_DIR_NAME, prefix="val")
            make_json_file(img_data, ann_data, cats, prefix="val")
            img_data=[]
            ann_data=[]
            print("Done")
        elif img_count == all_img_num:
            print("Make Train File...")
            make_img_file(img_data, IMAGE_DIR, ORIG_DIR_NAME, prefix="train")
            make_json_file(img_data, ann_data, cats, prefix="train")
            img_data=[]
            ann_data=[]
            print("Done")
    






