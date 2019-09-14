from PACK import *

# from aggregator import MeanAggregator
from aggregator_with_weight import MeanAggregator

class Encoder(nn.Module):
    '''
    encodes a node's embedding using GraphSAGE approach
    '''

    def __init__(self, embedding, feature_dim, embed_dim, adj_lists, num_sample=10, gcn=False, use_cuda=False):
        super(Encoder, self).__init__()
        self.embedding = embedding
        self.gcn = gcn
        self.use_cuda = use_cuda
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.adj_lists = adj_lists
        self.num_sample = num_sample
        self.aggregator = MeanAggregator(self.embedding, self.gcn, self.use_cuda)

        self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim, self.feature_dim if self.gcn else 2*self.feature_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        '''
        nodes: list of nodes in a batch
        '''
        embedded_features = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], self.num_sample)
        if not self.gcn:
            if self.use_cuda:
                self_feats = self.embedding(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.embedding(torch.LongTensor(nodes))
            encoding = torch.cat((self_feats, embedded_features), dim=1)
        else:
            encoding = embedded_features
        new_feature = self.weight.mm(encoding.t())
        new_feature = F.normalize(new_feature, p=2, dim=0)
        return new_feature

    def queryForward(self, query_feature):
        '''
        online query embedding inference.
        '''
        embedded_features = query_feature
        if not self.gcn:
            encoding = torch.cat((query_feature, embedded_features), dim=1)
        else:
            encoding = embedded_features
        new_feature = self.weight.mm(encoding.t())

        new_feature = F.normalize(new_feature, p=2, dim=0)
        return new_feature.t()
