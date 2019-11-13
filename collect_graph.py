import os
import shutil

from collections import defaultdict

import numpy as np

def removeIsolated(suffix = '.f.npy'):
    '''
    remove isolated points which have no neighbors.
    return number of nodes and categories.
    '''
    src_folder = '/path/to/train_feature'
    class_folders = os.listdir(src_folder)
    node_num = 0
    class_num = 0
    for c, folder in enumerate(class_folders):
        path = os.path.join(src_folder, folder)
        item_num = len(filter(lambda f: f.endswith(suffix), os.listdir(path)))
        if item_num > 1:
            node_num += item_num
            class_num += 1
        else:
            print "remove class-{}".format(c)
            shutil.rmtree(path)

    print "node num.:", node_num
    print "class num.:", class_num

    return node_num, class_num

def collectGraph_train(node_num, class_num, feat_dim = 256, suffix = '.f.npy'):
    '''
    (training dataset)
    collect info. about graph including: node, label, feature, neighborhood(adjacent) relationship.
    neighborhood(adjacent) relationship are constructed based on label.
    '''

    label = np.empty((node_num,1), dtype=np.int64)
    feature_map = np.zeros((node_num, feat_dim))
    adj_lists = defaultdict(set)

    src_folder = '/path/to/train_feature'
    class_folders = os.listdir(src_folder)
    idx = 0
    for c, folder in enumerate(class_folders):
        # print c
        path = os.path.join(src_folder, folder)
        # print path
        items = filter(lambda f: f.endswith(suffix), os.listdir(path))
        item_num = len(items)
        for i in range(item_num):
            item = items[i]
            ind_i = i + idx
            feature = np.load(os.path.join(path, item))
            feature_map[ind_i,:] = feature
            label[ind_i] = c
            for j in range(i+1, item_num):
                ind_j = j + idx
                adj_lists[ind_i].add(ind_j)
                adj_lists[ind_j].add(ind_i)
        idx += item_num

    return label, feature_map, adj_lists

def collectGraph_train_v2(node_num, class_num, feat_dim = 256, knn = 10, suffix = '.f.npy', round="0"):
    '''
    (training dataset)
    collect info. about graph including: node, label, feature, neighborhood(adjacent) relationship.
    neighborhood(adjacent) relationship are constructed based on similarity between features.
    '''

    feature_map = np.load('/path/to/feature_map_{}.npy'.format(round))
    label = np.load('/path/to/label.npy')

    similarity = np.dot(feature_map, feature_map.T)
    sort_id = np.argsort(-similarity, axis=1)
    for n in range(node_num):
        similarity[n, sort_id[n, knn+1:]] = 0
        similarity[n] /= np.sum(similarity[n])
        # similarity[n, sort_id[n, knn+1:]] = -9e10
        # similarity[n] = np.exp(similarity[n])
        # similarity[n] /= np.sum(similarity[n])

    adj_lists = defaultdict(set)

    for n in range(node_num):
        for k in range(1, knn+1):
            adj_lists[n].add((sort_id[n,k], similarity[n, sort_id[n][k]]))

    # feature_map = np.load('/data4/fong/pytorch/Graph/train_feature_map/feature_map_0.npy')

    return label, feature_map, adj_lists

def collectGraph_test(feature_path, node_num, feat_dim = 256, knn = 10, suffix = '.f.npy', round="0"):
    print "node num.:", node_num

    feature_map = np.load(os.path.join(feature_path, 'feature_map_{}.npy'.format(round)))
    similarity = np.dot(feature_map, feature_map.T)
    sort_id = np.argsort(-similarity, axis=1)
    for n in range(node_num):
        similarity[n, sort_id[n, knn+1:]] = 0
        similarity[n] /= np.sum(similarity[n])
        # similarity[n, sort_id[n, knn+1:]] = -9e10
        # similarity[n] = np.exp(similarity[n])
        # similarity[n] /= np.sum(similarity[n])

    adj_lists = defaultdict(set)

    for n in range(node_num):
        for k in range(1, knn+1):
            adj_lists[n].add((sort_id[n,k], similarity[n, sort_id[n][k]]))

    return feature_map, adj_lists

if __name__ == "__main__":
    node_num, class_num = removeIsolated(suffix = '.f.npy')
    node_num, class_num = removeIsolated(suffix = '.fr.npy')
    node_num, class_num = removeIsolated(suffix = '.frmac.npy')
