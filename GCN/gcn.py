import torch
from torch.nn import ReLU, Dropout
from torch_geometric.nn import global_max_pool
from gcn_layer import CNN_Layer


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = CNN_Layer(9, 16)
        self.conv2 = CNN_Layer(16, 1)
        # self.conv3 = CNN_Layer(32, 1)
        self.relu = ReLU()
        self.dropout = Dropout(0.4)
        self.pool = global_max_pool

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.conv3(x, edge_index, edge_attr)
        x_graph = self.pool(x, data.batch)

        return x_graph
