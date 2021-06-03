import torch
from torch.nn import ReLU, Dropout
from torch_geometric.nn import global_max_pool, global_add_pool
from gat_layer import GAT_Layer
from ogb.graphproppred.mol_encoder import AtomEncoder


class Net(torch.nn.Module):
    def __init__(self, emb_dim=300, num_tasks=1):
        super(Net, self).__init__()
        self.conv1 = GAT_Layer(emb_dim)
        self.conv2 = GAT_Layer(emb_dim)
        self.relu = ReLU()
        self.dropout = Dropout(0.4)
        self.pool = global_add_pool
        self.atom_encoder = AtomEncoder(emb_dim)
        self.lin_graph_pred = torch.nn.Linear(emb_dim, num_tasks)

    def forward(self, data, eval=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.atom_encoder(x)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        if eval == False:
            x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.pool(x, data.batch)
        x = self.lin_graph_pred(x)

        return x
