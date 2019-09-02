from PACK import *
from torch.optim.lr_scheduler import StepLR

from encoder import Encoder
from model import UnsupervisedGraphSAGE_Single
from utils import buildTestData
from collect_graph import removeIsolated, collectGraph_train, collectGraph_train_v2, collectGraph_test

import numpy as np
import math
import time
import random
import visdom
from tqdm import tqdm
from copy import deepcopy

import argparse
import ast

import sys

eval_func = '/data4/fong/oxford5k/evaluation/compute_ap'
retrieval_result = '/data4/fong/pytorch/Graph/retrieval'
test_dataset = {
    'oxf': {
        'node_num': 5063,
        'img_testpath': '/data4/fong/pytorch/RankNet/building/test_oxf/images',
        'feature_path': '/data4/fong/pytorch/Graph/test_feature_map/oxford',
        'gt_path': '/data4/fong/oxford5k/oxford5k_groundTruth',
    },
    'par': {
        'node_num': 6392,
        'img_testpath': '/data4/fong/pytorch/RankNet/building/test_par/images',
        'feature_path': '/data4/fong/pytorch/Graph/test_feature_map/paris',
        'gt_path': '/data4/fong/paris6k/paris_groundTruth',
    }
}
building_oxf = buildTestData(img_path=test_dataset['oxf']['img_testpath'], gt_path=test_dataset['oxf']['gt_path'], eval_func=eval_func)
building_par = buildTestData(img_path=test_dataset['par']['img_testpath'], gt_path=test_dataset['par']['gt_path'], eval_func=eval_func)
building = {
    'oxf': building_oxf,
    'par': building_par,
}

def makeModel(node_num, class_num, feature_map, adj_lists, args):
    ## feature embedding
    embedding = nn.Embedding(node_num, args.feat_dim)
    embedding.weight = nn.Parameter(torch.from_numpy(feature_map).float(), requires_grad=False)

    ## single-layer encoder
    encoder = Encoder(embedding, args.feat_dim, args.embed_dim, adj_lists, num_sample=args.num_sample, gcn=args.use_gcn, use_cuda=args.use_cuda)

    ## model
    graphsage = UnsupervisedGraphSAGE_Single(class_num, encoder)
    if args.use_cuda:
        embedding.cuda()
        encoder.cuda()
        graphsage.cuda()

    return graphsage

def train(args):
    ## load training data
    print "loading training data ......"
    node_num, class_num = removeIsolated(args.suffix)
    # node_num, class_num = 33792, 569
    # label, feature_map, adj_lists = collectGraph_train(node_num, class_num, args.feat_dim, args.suffix)
    label, feature_map, adj_lists = collectGraph_train_v2(node_num, class_num, args.feat_dim, args.num_sample, args.suffix)

    graphsage = makeModel(node_num, class_num, feature_map, adj_lists, args)

    optimizer = torch.optim.Adam(filter(lambda para: para.requires_grad, graphsage.parameters()), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # for name, para in graphsage.named_parameters():
    #     print name
    #     print para.requires_grad
    #     print para
    # sys.exit(0)
    # print optimizer.param_groups
    # sys.exit(0)

    ## train
    random.seed(2)
    train_nodes = range(args.train_num)
    positive = [random.choice(filter(lambda node: node != anchor and label[node] == label[anchor], train_nodes)) for anchor in train_nodes]

    epoch_num = args.epoch_num
    batch_size = args.batch_size
    iter_num = int(math.ceil(args.train_num / float(batch_size)))
    check_loss = []
    check_step = args.check_step
    train_loss = 0.0
    iter_cnt = 0

    graphsage.train()
    for e in range(epoch_num):

        scheduler.step()

        random.shuffle(train_nodes)
        for batch in range(iter_num):
            anchor_nodes = train_nodes[batch*batch_size: (batch+1)*batch_size]
            positive_nodes = [positive[anchor] for anchor in anchor_nodes]
            anchor_label = torch.LongTensor(label[anchor_nodes])
            if args.use_cuda:
                anchor_label = anchor_label.cuda()
            optimizer.zero_grad()
            loss = graphsage.loss(anchor_nodes, positive_nodes, anchor_label)
            loss.backward()
            optimizer.step()
            iter_cnt += 1
            train_loss += loss.cpu().item()
            if iter_cnt % check_step == 0:
                check_loss.append(train_loss/check_step)
                print time.strftime('%Y-%m-%d %H:%M:%S'), "epoch: {}, iter: {}, loss:{:.4f}".format(e, iter_cnt, train_loss/check_step)
                train_loss = 0.0


    checkpoint_path = 'checkpoint/checkpoint_single_unsup_{}.pth'.format(time.strftime('%Y%m%d%H%M'))
    torch.save({
            'train_num': args.train_num,
            'epoch_num': args.epoch_num,
            'learning_rate': args.learning_rate,
            'embed_dim': args.embed_dim,
            'num_sample': args.num_sample,
            'use_gcn': args.use_gcn,
            'graph_state_dict': graphsage.state_dict(),
            'optimizer': optimizer.state_dict(),
            },
            checkpoint_path)

    vis = visdom.Visdom(env='Graph', port='8099')
    vis.line(
            X = np.arange(1, len(check_loss)+1, 1) * check_step,
            Y = np.array(check_loss),
            opts = dict(
                title=time.strftime('%Y-%m-%d %H:%M:%S') + ', single_unsup, gcn {}'.format(args.use_gcn),
                xlabel='itr.',
                ylabel='loss'
            )
    )

    return checkpoint_path, class_num

def test(checkpoint_path, class_num, args):
    for key in building.keys():
        node_num = test_dataset[key]['node_num']
        old_feature_map, adj_lists = collectGraph_test(test_dataset[key]['feature_path'], node_num, args.feat_dim, args.num_sample, args.suffix)

        graphsage = makeModel(node_num, class_num, old_feature_map, adj_lists, args)

        checkpoint = torch.load(checkpoint_path)
        graphsage_state_dict = graphsage.state_dict()
        graphsage_state_dict.update({'weight': checkpoint['graph_state_dict']['weight']})
        graphsage_state_dict.update({'encoder.weight': checkpoint['graph_state_dict']['encoder.weight']})
        graphsage.load_state_dict(graphsage_state_dict)
        graphsage.eval()

        batch_num = int(math.ceil(node_num/float(args.batch_size)))
        new_feature_map = torch.FloatTensor()
        for batch in tqdm(range(batch_num)):
            start_node = batch*args.batch_size
            end_node = min((batch+1)*args.batch_size, node_num)
            test_nodes = range(start_node, end_node)
            new_feature = graphsage(test_nodes)
            new_feature = F.normalize(new_feature, p=2, dim=0)
            new_feature_map = torch.cat((new_feature_map, new_feature.t().cpu().data), dim=0)
        new_feature_map = new_feature_map.numpy()
        old_similarity = np.dot(old_feature_map, old_feature_map.T)
        new_similarity = np.dot(new_feature_map, new_feature_map.T)
        mAP_old = building[key].evalRetrieval(old_similarity, retrieval_result)
        mAP_new = building[key].evalRetrieval(new_similarity, retrieval_result)
        print time.strftime('%Y-%m-%d %H:%M:%S'), 'eval {}'.format(key)
        print 'base feature: {}, new feature: {}'.format(old_feature_map.shape, new_feature_map.shape)
        print 'base mAP: {:.4f}, new mAP: {:.4f}, improve: {:.4f}'.format(mAP_old, mAP_new, mAP_new-mAP_old)

        meanAggregator = graphsage.encoder.aggregator
        ## directly update node's features by mean pooling features of its neighbors.
        mean_feature_map = torch.FloatTensor()
        for batch in tqdm(range(batch_num)):
            start_node = batch*args.batch_size
            end_node = min((batch+1)*args.batch_size, node_num)
            test_nodes = range(start_node, end_node)
            mean_feature = meanAggregator(test_nodes, [adj_lists[i] for i in test_nodes], args.num_sample)
            mean_feature = F.normalize(mean_feature, p=2, dim=1)
            mean_feature_map = torch.cat((mean_feature_map, mean_feature.cpu().data), dim=0)
        mean_feature_map = mean_feature_map.numpy()
        mean_similarity = np.dot(mean_feature_map, mean_feature_map.T)
        mAP_mean = building[key].evalRetrieval(mean_similarity, retrieval_result)
        print 'mean aggregation mAP: {:.4f}'.format(mAP_mean)
        print ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Unsupervised Single-layer GraphSAGE, train on Landmark_clean, test on Oxford5k and Paris6k.')
    parser.add_argument('-E', '--epoch_num', type=int, default=70, required=False, help='training epoch number.')
    parser.add_argument('-R', '--step_size', type=int, default=30, required=False, help='learning rate decay step_size.')
    parser.add_argument('-B', '--batch_size', type=int, default=64, required=False, help='training batch size.')
    parser.add_argument('-S', '--check_step', type=int, default=50, required=False, help='loss check step.')
    parser.add_argument('-C', '--use_cuda', type=ast.literal_eval, default=True, required=False, help='whether to use gpu (True) or not (False).')
    parser.add_argument('-G', '--use_gcn', type=ast.literal_eval, default=True, required=False, help='whether to use gcn (True) or not (False).')
    parser.add_argument('-L', '--learning_rate', type=float, default=0.005, required=False, help='training learning rate.')
    parser.add_argument('-N', '--num_sample', type=int, default=10, required=False, help='number of neighbors to aggregate.')
    parser.add_argument('-x', '--suffix', type=str, default='.frmac.npy', required=False, help='feature type, \'f\' for vggnet (512-d), \'fr\' for resnet (2048-d), \'frmac\' for vgg16_rmac (512-d).')
    parser.add_argument('-f', '--feat_dim', type=int, default=512, required=False, help='input feature dim of node.')
    parser.add_argument('-d', '--embed_dim', type=int, default=512, required=False, help='embedded feature dim of encoder.')
    parser.add_argument('-T', '--train_num', type=int, default=33792, required=False, help='number of training nodes (less than 36460). Left for validation.')
    args, _ = parser.parse_known_args()
    print "< < < < < < < Unsupervised Single-layer GraphSAGE > > > > > > >"
    print "= = = = = = = = = = = PARAMETERS SETTING = = = = = = = = = = ="
    print "epoch_num:", args.epoch_num
    print "step_size:", args.step_size
    print "batch_size:", args.batch_size
    print "check_step:", args.check_step
    print "train_num:", args.train_num
    print "learning_rate:", args.learning_rate
    print "suffix:", args.suffix
    print "feat_dim:", args.feat_dim
    print "embed_dim:", args.embed_dim
    print "num_sample:", args.num_sample
    print "use_cuda:", args.use_cuda
    print "use_gcn:", args.use_gcn
    print "= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="

    print "training ......"
    checkpoint_path, class_num = train(args)

    print "testing ......"
    # checkpoint_path = 'checkpoint/checkpoint_single_unsup_201904291614.pth'
    # class_num = 569
    test(checkpoint_path, class_num, args)
