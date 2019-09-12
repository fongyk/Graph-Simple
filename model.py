from PACK import *

## supervised graphsage, classification loss
class SupervisedGraphSAGE(nn.Module):
    def __init__(self, class_num, encoder_1, encoder_2):
        super(SupervisedGraphSAGE, self).__init__()
        self.class_num = class_num
        self.encoder_1 = encoder_1
        self.encoder = encoder_2
        self.criterion = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(class_num, self.encoder.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        new_feature = self.encoder(nodes)
        scores = self.weight.mm(new_feature)
        return new_feature, scores.t()

    def loss(self, nodes, labels):
        _, scores = self.forward(nodes)
        return self.criterion(scores, labels.squeeze())

class SupervisedGraphSAGE_Single(nn.Module):
    def __init__(self, class_num, encoder):
        super(SupervisedGraphSAGE_Single, self).__init__()
        self.class_num = class_num
        self.encoder = encoder
        self.criterion = nn.CrossEntropyLoss()

        self.fc = nn.Sequential(
                nn.Linear(encoder.embed_dim, class_num, bias=True)
        )

    def forward(self, nodes):
        new_feature = self.encoder(nodes)
        hidden_activation = F.relu(new_feature)
        scores = self.fc(hidden_activation.t())
        return new_feature, scores

    def loss(self, nodes, labels):
        _, scores = self.forward(nodes)
        return self.criterion(scores, labels.squeeze())


## unsupervised graphsage, n-pair loss
class UnsupervisedGraphSAGE_Single(nn.Module):
    def __init__(self, class_num, encoder):
        super(UnsupervisedGraphSAGE_Single, self).__init__()
        self.class_num = class_num
        self.encoder = encoder
        self.criterion = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(class_num, encoder.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        return self.encoder(nodes)

    def loss(self, nodes_anchor, nodes_positive, label):

        ## triplet loss + classification loss
        anchor_feature = self.forward(nodes_anchor)
        positive_feature = self.forward(nodes_positive)
        anchor_activation = F.relu(anchor_feature)
        positive_activation = F.relu(positive_feature)
        anchor_score = self.weight.mm(anchor_activation)
        positive_score = self.weight.mm(positive_activation)
        target = label.view(label.size(0), -1)
        target = (target != target.t()).float()

        sqdist = 2 - 2 * torch.matmul(anchor_feature.t(), positive_feature)
        pos_dist = torch.diagonal(sqdist)
        diff_dist = pos_dist.view(-1, 1).repeat(1, sqdist.size(0)) - sqdist
        loss_feature = torch.mean(torch.sum(F.relu(diff_dist + 0.3) * target, dim=1))
        loss_class = self.criterion(anchor_score.t(), label.squeeze()) + self.criterion(positive_score.t(), label.squeeze())

        return loss_feature + loss_class * 0.5

        ## n-pair loss + classification loss

        # anchor_feature = self.forward(nodes_anchor)
        # positive_feature = self.forward(nodes_positive)
        # anchor_score = self.weight.mm(anchor_feature)
        # positive_score = self.weight.mm(positive_feature)
        #
        # anchor_feature = F.normalize(anchor_feature, p=2, dim=0)
        # positive_feature = F.normalize(positive_feature, p=2, dim=0)
        # target = label.view(label.size(0), -1)
        # target = (target == target.t()).float()
        #
        # target = target / torch.sum(target, dim=1, keepdim=True).float()
        # logit = torch.matmul(anchor_feature.t(), positive_feature)
        #
        # loss_feature = - torch.mean(torch.sum(target * F.log_softmax(logit, dim=1), dim=1))
        # loss_class = self.criterion(anchor_score.t(), label.squeeze()) + self.criterion(positive_score.t(), label.squeeze())
        #
        # return loss_class + 0.05 * loss_feature
