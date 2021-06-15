import torch
from torch.nn import ReLU, Dropout
from torch_geometric.nn import global_max_pool, global_add_pool
from gin_layer import GNN_Node, GNN_Node_Virtualnode
from ogb.graphproppred.mol_encoder import AtomEncoder


class Net(torch.nn.Module):
    def __init__(
        self,
        num_tasks=1,
        num_layers=5,
        emb_dim=300,
        gnn_type="gin",
        residual=False,
        drop_ratio=0,
        JK="last",
        graph_pooling="sum",
        virtual_node=False
    ):
        super(Net, self).__init__()
        if virtual_node:
            self.gnn_node = GNN_Node_Virtualnode(num_layers, emb_dim, drop_ratio, JK, residual, gnn_type)
        else:
            self.gnn_node = GNN_Node(gnn_type, emb_dim, num_layers, drop_ratio, residual, JK)
        self.lin_graph_pred = torch.nn.Linear(emb_dim, num_tasks)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool

    def forward(self, data):
        n_emb = self.gnn_node(data)
        g_emb = self.pool(n_emb, data.batch)
        out = self.lin_graph_pred(g_emb)
        if not self.training:
            out = torch.clamp(out, min=0, max=50)

        return out
