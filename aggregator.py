from PACK import *

import random

class MeanAggregator(nn.Module):
    '''
    aggregate a node's feature using mean of features of its neighbors
    '''

    def __init__(self, embedding, gcn=False, use_cuda=False):
        '''
        (torch.nn.Embedding)
        embedding is a look up table that stores features of all nodes,
        that maps LongTensor node ids to FloatTensor of feature values.

        gcn=False: perform concatenation GraphSAGE-style
        gcn=True: add self-loops (GCN style).
        '''
        super(MeanAggregator, self).__init__()
        self.embedding = embedding
        self.gcn = gcn
        self.use_cuda = use_cuda

    def forward(self, nodes, neighbors, num_sample=10):
        '''
        nodes: list of nodes in a batch
        neighbors: list of sets, each set includes neighbors of each node
        num_sample: number of neighbors to sample. No sampling if None
        '''
        _set = set
        if num_sample is not None:
            _sample = random.sample
            sample_neighbors = [_set(_sample(neigh, num_sample)) if len(neigh)>=num_sample else neigh for neigh in neighbors]
        else:
            sample_neighbors = neighbors
        if self.gcn:
            sample_neighbors = [(sample_neigh | set([nodes[i]])) for i, sample_neigh in enumerate(sample_neighbors)]
        unique_nodes_list = list(set.union(*sample_neighbors))
        unique_nodes = {node:i for i,node in enumerate(unique_nodes_list)}

        ## mask[i,j] = 1 means that j-th node appears in i-th neighbor set
        mask = Variable(torch.zeros(len(sample_neighbors), len(unique_nodes)))
        row_indices = [k for k in range(len(sample_neighbors)) for i in range(len(sample_neighbors[k]))]
        col_indices = [unique_nodes[node] for sample_neigh in sample_neighbors for node in sample_neigh]
        mask[row_indices, col_indices] = 1
        if self.use_cuda:
            mask = mask.cuda()
        neighbor_num = mask.sum(1, keepdim=True)
        mask = mask.div(neighbor_num)
        if self.use_cuda:
            features_unique_nodes = self.embedding(torch.LongTensor(unique_nodes_list).cuda())
        else:
            features_unique_nodes = self.embedding(torch.LongTensor(unique_nodes_list))
        embedded_features = mask.mm(features_unique_nodes)

        return embedded_features
