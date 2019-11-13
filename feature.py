import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import numpy as np
from sklearn import preprocessing
from tqdm import tqdm
import struct

from collections import defaultdict

import os
import sys
import scipy.io as sio

img_transform = transforms.Compose([
    transforms.Resize(480),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_feature_path = "train_feature"
train_path = "/path/to/landmark_clean"
train_data = ImageFolder(train_path, transform=img_transform)
train_dataloader = DataLoader(dataset=train_data, shuffle=False, num_workers=4, batch_size=1)

test_dataset = {
    'oxf': {
        'img_path': '/path/to/images',
        'feature_path': '/path/to/feature',
    },
    'par':{
        'img_path': '/path/to/images',
        'feature_path': '/path/to/feature',
    }
}
test_data_oxf = ImageFolder(test_dataset['oxf']['img_path'], transform=img_transform)
test_dataloader_oxf = DataLoader(dataset=test_data_oxf, shuffle=False, num_workers=4, batch_size=1)
test_data_par = ImageFolder(test_dataset['par']['img_path'], transform=img_transform)
test_dataloader_par = DataLoader(dataset=test_data_par, shuffle=False, num_workers=4, batch_size=1)
test_dataloader = {
    'oxf': test_dataloader_oxf,
    'par': test_dataloader_par,
}

class AlexNetFeature(nn.Module):
    def __init__(self):
        super(AlexNetFeature, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        self.feature = nn.Sequential(*(alexnet.features[i] for i in range(12)))
        self.feature.add_module('12: Global Pooling', nn.AdaptiveMaxPool2d(1))

    def forward(self, I):
        f = self.feature(I)
        f = f.view(f.size(0), -1)
        f = F.normalize(f, p=2, dim=1)
        return f

class VGGNetFeature(nn.Module):
    def __init__(self):
        super(VGGNetFeature, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.feature = nn.Sequential(*(vgg.features[i] for i in range(36)))
        self.feature.add_module('36: GlobalPooling', nn.AdaptiveMaxPool2d(1))

    def forward(self, I):
        f = self.feature(I)
        f = f.view(f.size(0), -1)
        f = F.normalize(f, p=2, dim=1)
        return f

class ResNetFeature(nn.Module):
    def __init__(self):
        super(ResNetFeature, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature = nn.Sequential(*list(resnet.children())[:-2])
        self.feature.add_module('8: Global Pooling', nn.AdaptiveMaxPool2d(1))

    def forward(self, I):
        f = self.feature(I)
        f = f.view(f.size(0), -1)
        f = F.normalize(f, p=2, dim=1)
        return f

def extractFeature(model, data_loader, feature_path, write_flag=False, suffix='.f.npy'):
    '''
    write feature to .npy file or return feature matrix.
    '''

    model.cuda()
    model.eval()
    if write_flag == True:
        cnt = 0
        for img, label in data_loader:
            img = Variable(img).cuda()
            feature = model(img)
            class_path = os.path.join(feature_path, str(int(label)))
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            save_dir = os.path.join(class_path, str(cnt) + suffix)
            print save_dir
            np.save(save_dir, feature.cpu().data.numpy())
            cnt += 1
    else:
        feature_map = torch.FloatTensor()
        for img, _ in tqdm(data_loader):
            img = Variable(img).cuda()
            feature = model(img)
            feature_map = torch.cat((feature_map, feature.cpu().data), 0)
        feature_map = feature_map.numpy()
        return feature_map

def proceeRMACfeature(img_path, src_path, dest_path):
    '''
    vgg16_rmac is a 1024-d vector, composed of 512-d max-pooling feature and 512-d rmac feature.
    '''
    folders = os.listdir(img_path)
    folders.sort()
    cnt = 0
    for folder in folders:
        folder_path = os.path.join(img_path, folder)
        images = os.listdir(folder_path)
        images.sort()
        for img in images:
            src_feature = os.path.join(src_path, img+'.vgg16_rmac')
            if not os.path.exists(src_feature):
                continue
            print src_feature
            fr = open(src_feature, 'rb')
            feat_dim = 512
            f_max = struct.unpack('f'*feat_dim, fr.read(4*feat_dim))
            # f_max = np.array(f_max)
            # f_max = f_max.reshape(1, -1)
            # f_max = preprocessing.normalize(f_max, norm='l2', axis=1)
            f_rmac = struct.unpack('f'*feat_dim, fr.read(4*feat_dim))
            f_rmac = np.array(f_rmac)
            f_rmac = f_rmac.reshape(1, -1)
            f_rmac = preprocessing.normalize(f_rmac, norm='l2', axis=1)
            feature_folder = os.path.join(dest_path, folder)
            if not os.path.exists(feature_folder):
                os.makedirs(feature_folder)
            dest_feature = os.path.join(feature_folder, str(cnt) + '.frmac.npy')
            # np.save(dest_feature, np.column_stack((f_max, f_rmac)))
            np.save(dest_feature, f_rmac)
            fr.close()
            cnt += 1
    print "number of features:", cnt

def loadRMACfeatureFromMat(feature_path, matFile):
    '''
    load 512-d vgg16_rmac pca-whitened feature from .mat file
    '''

    mat = sio.loadmat(os.path.join(feature_path, matFile))
    database_feature = mat['feat']
    query_feature = mat['qfeat']
    query_idx = mat['qidx']
    query_idx = query_idx.reshape(55)

    np.save(os.path.join(feature_path, 'database_feature.npy'), database_feature.T)
    np.save(os.path.join(feature_path, 'query_feature.npy'), query_feature.T)
    np.save(os.path.join(feature_path, 'query_idx.npy'), query_idx)

if __name__ == "__main__":
    model, suffix = AlexNetFeature(), '.f.npy'
    for building, building_dataloader in test_dataloader.items():
        extractFeature(model, building_dataloader, test_dataset[building]['feature_path'], True, suffix)