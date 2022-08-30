## 目的
重機姿勢特定

## 実行環境
* Ubuntu 18.04LTS
* Python >= 3.6

## Installation
1.自分の環境にあったバージョンのpytorchを公式サイト(https://pytorch.org)からインストール　　

2.それ以外のモジュールを下記コードを実行してインストール
```bash
pip3 install -r requirements.txt
```

## 学習・推論の実行方法
学習
```bash
python3 train_VGG19.py
```
推論
```bash
cd lib/pafprocess
sudo apt install swig
sh make.sh
cd ../../
python3 picture_demo.py
```

## 参考コード
OpenPose:
https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation

## Citation
Please cite the paper in your publications if it helps your research: 

    @InProceedings{cao2017realtime,
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2017}
      }
