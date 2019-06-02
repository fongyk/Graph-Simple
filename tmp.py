import torch
import torch.nn as nn

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix
    def __getitem__(self, index):
        return getattr(self.module, self.prefix + str(index))

class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.layerList = [nn.Linear(5, 1, bias=False), nn.Linear(2,1, bias=False)]
        self.add_module("layer_0", self.layerList[0])
        self.add_module("layer_1", self.layerList[1])

        self.layer = AttrProxy(self, "layer_")

        print self.layerList[0].weight
        print self.layer[0].weight
        self.layer[0].weight.data = self.layer[0].weight.data + 2 * torch.ones_like(self.layer[0].weight.data)

    def forward(self):
        print self.layerList[0].weight
        print self.layer[0].weight
        print self.layer[1].weight
