from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.nn import init

#VGG19のブロックを作成する
def make_vgg19_block(block):
    """Builds a vgg19 block from a dictionary
    Args:
        block: a dictionary
    """
    layers = []
    for i in range(len(block)):
        one_ = block[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

class ConvBlock(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
        self.conv2 = nn.Conv2d(in_channels=v[1], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
        self.conv3 = nn.Conv2d(in_channels=v[1], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.relu1(out1)
        out2 = self.conv2(out1)
        out2 = self.relu2(out2)
        out3 = self.conv3(out2)
        out3 = self.relu3(out3)
        out = torch.cat([out1, out2, out3], 1)
        return out
    
#Conv層の作成を簡略化
def make_conv_stage(cfg_dict):
    layers = []
    for i in range(len(cfg_dict)-1):
        one_ = cfg_dict[i]
        for k, v in one_.items():
            if 'block' in k:
                conv_block = ConvBlock(v)
                layers += [conv_block]
            else:    
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                kernel_size=v[2], stride=v[3], padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                       kernel_size=v[2], stride=v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)

#modelの構造を記述する(辞書形式)
def get_model(trunk="vgg", paf_stage_num=4, cm_stage_num=2):
    #ブロックを初期化する
    paf_blocks = OrderedDict()
    cm_blocks = OrderedDict()
    #VGG19を構成するための情報
    vgg_block =  [{'conv1_1': [3, 64, 3, 1, 1]},
                  {'conv1_2': [64, 64, 3, 1, 1]},
                  {'pool1_stage1': [2, 2, 0]},
                  {'conv2_1': [64, 128, 3, 1, 1]},
                  {'conv2_2': [128, 128, 3, 1, 1]},
                  {'pool2_stage1': [2, 2, 0]},
                  {'conv3_1': [128, 256, 3, 1, 1]},
                  {'conv3_2': [256, 256, 3, 1, 1]},
                  {'conv3_3': [256, 256, 3, 1, 1]},
                  {'conv3_4': [256, 256, 3, 1, 1]},
                  {'pool3_stage1': [2, 2, 0]},
                  {'conv4_1': [256, 512, 3, 1, 1]},
                  {'conv4_2': [512, 512, 3, 1, 1]},
                  {'conv4_3_CPM': [512, 256, 3, 1, 1]},
                  {'conv4_4_CPM': [256, 128, 3, 1, 1]}]
    #pafブロックの1ステージ目
    paf_blocks["paf_block1"] = [
        {"conv_block_1": [128, 64, 3, 1, 1]},
        {"conv_block_2": [192, 64, 3, 1, 1]},
        {"conv_block_3": [192, 64, 3, 1, 1]},
        {"conv_block_4": [192, 64, 3, 1, 1]},
        {"conv_block_5": [192, 64, 3, 1 ,1]},
        {"conv_layer_6": [192, 512, 1, 1, 0]},
        {"conv_layer_7": [512, 22, 1, 1, 0]}
    ]
    #confidence mapの1ステージ目
    cm_blocks["cm_block1"] = [
        {"conv_block_1": [150, 64, 3, 1, 1]},
        {"conv_block_2": [192, 64, 3, 1, 1]},
        {"conv_block_3": [192, 64, 3, 1, 1]},
        {"conv_block_4": [192, 64, 3, 1, 1]},
        {"conv_block_5": [192, 64, 3, 1, 1]},
        {"conv_layer_6": [192, 512, 1, 1, 0]},
        {"conv_layer_7": [512, 13, 1, 1, 0]}
    ]
    #pafとconfidence mapの2層目以降の情報を記述
    for i in range(2, paf_stage_num+1):
        paf_blocks["paf_block%d" %i] = [
            {"conv_block_1": [150, 64, 3, 1, 1]},
            {"conv_block_2": [192, 64, 3, 1, 1]},
            {"conv_block_3": [192, 64, 3, 1, 1]},
            {"conv_block_4": [192, 64, 3, 1, 1]},
            {"conv_block_5": [192, 64, 3, 1 ,1]},
            {"conv_layer_6": [192, 512, 1, 1, 0]},
            {"conv_layer_7": [512, 22, 1, 1, 0]}
        ]
    for i in range(2, cm_stage_num+1):
        cm_blocks["cm_block%d" %i] = [
            {"conv_block_1": [163, 64, 3, 1, 1]},
            {"conv_block_2": [192, 64, 3, 1, 1]},
            {"conv_block_3": [192, 64, 3, 1, 1]},
            {"conv_block_4": [192, 64, 3, 1, 1]},
            {"conv_block_5": [192, 64, 3, 1, 1]},
            {"conv_layer_6": [192, 512, 1, 1, 0]},
            {"conv_layer_7": [512, 13, 1, 1, 0]}
        ]

    #モデルの情報に変換する
    paf_models = OrderedDict()
    cm_models = OrderedDict()
    #1層目(vgg block)
    paf_models["block0"] = make_vgg19_block(vgg_block)
    #2層目以降
    for paf in paf_blocks.items():
        paf_models[paf[0]] = make_conv_stage(paf[1])
    for cm in cm_blocks.items():
        cm_models[cm[0]] = make_conv_stage(cm[1])  
    

    #改良されたOpenPoseの構造を記述
    class MyRtpose(nn.Module):
        def __init__(self, paf_models, cm_models):
            super().__init__()
            self.model0 = paf_models["block0"]
            self.paf_models = nn.ModuleList(paf_models["paf_block%d" %i] for i in range(1, len(paf_models)))
            self.cm_models = nn.ModuleList(cm_models["cm_block%d" %i] for i in range(1, len(cm_models)+1))

            self._initialize_weights_norm()
        
        def forward(self, x):
            saved_for_loss = []
            F = self.model0(x)
            paf_in = F
            for paf_model in self.paf_models:
                out = paf_model(paf_in)
                paf_in = torch.cat([F, out], 1)
                saved_for_loss.append(out)
            
            paf_out = out
            cm_in = torch.cat([F, paf_out], 1)
            for cm_model in self.cm_models:
                out = cm_model(cm_in)
                cm_in = torch.cat([F, paf_out, out], 1)
                saved_for_loss.append(out)

            cm_out = out
            return (paf_out, cm_out), saved_for_loss
                  
        def _initialize_weights_norm(self):

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.normal_(m.weight, std=0.01)
                    if m.bias is not None:  # mobilenet conv2d doesn't add bias
                        init.constant_(m.bias, 0.0)

            # last layer of these block don't have Relu
            for i in range(len(self.paf_models)):
                init.normal_(self.paf_models[i][-1].weight, std=0.01)
            
            for i in range(len(self.cm_models)):
                init.normal_(self.cm_models[i][-1].weight, std=0.01)

    model = MyRtpose(paf_models, cm_models)
    return model

def use_vgg(model):

    url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
    vgg_state_dict = model_zoo.load_url(url)
    vgg_keys = vgg_state_dict.keys()

    # load weights of vgg
    weights_load = {}
    # weight+bias,weight+bias.....(repeat 10 times)
    for i in range(20):
        weights_load[list(model.state_dict().keys())[i]
                     ] = vgg_state_dict[list(vgg_keys)[i]]

    state = model.state_dict()
    state.update(weights_load)
    model.load_state_dict(state)
    print('load imagenet pretrained model')
