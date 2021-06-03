# conda activate my-rdkit-env
from ogb.lsc import PygPCQM4MDataset, PCQM4MEvaluator
from ogb.utils import smiles2graph
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.data import DataLoader
import torch
from torch.nn import ReLU, Dropout
from torch_geometric.nn import (
    MessagePassing,
    global_max_pool,
    global_mean_pool,
    GCNConv,
)
from torch_geometric.utils import add_self_loops, degree
from tqdm import tqdm
from gin import Net

device = torch.device("cuda:5")

# smiles2graph takes a SMILES string as input and returns a graph object
# requires rdkit to be installed.
# You can write your own smiles2graph
# graph_obj = smiles2graph('O=C1C=CC(O1)C(c1ccccc1C)O')

# convert each SMILES string into a molecular graph object by calling smiles2graph
# This takes a while (a few hours) for the first run
dataset = PygPCQM4MDataset(
    root="/data3/kaleb.dickerson2001/Datasets/KDD-pyg-dataset",
    smiles2graph=smiles2graph,
)

split_idx = dataset.get_idx_split()
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)

atom_encoder = AtomEncoder(emb_dim=100)  # Pytorch Module class w/ learnable parameters
bond_encoder = BondEncoder(emb_dim=100)  # Pytorch Module class w/ learnable parameters


def train(model, loss, optimizer, epochs, save=True, scheduler=None):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Iteration {epoch}")):
            batch = batch.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(batch).view(-1,)
            # print(outputs)
            loss = criterion(outputs, batch.y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        if scheduler != None:
            scheduler.step()

    print("Finished Training")
    if save:
        torch.save(net.state_dict(), "GIN/Saves/GIN_10_epochs.pth")
        print("Saved")


def eval(model, evaluator):
    with torch.no_grad():
        y_true = []
        y_pred = []
        for data in tqdm(valid_loader, desc="Iteration"):
            data = data.to(device)
            outputs = model(data, eval=True).view(-1)
            y_true.append(data.y.view(outputs.shape).detach().cpu())
            y_pred.append(outputs.detach().cpu())

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        input_dict = {"y_true": y_true, "y_pred": y_pred}
        result_dict = evaluator.eval(input_dict)

    # Result
    print(result_dict["mae"])


net = Net()
net.to(device)
# net.load_state_dict(torch.load("GIN/Saves/GIN.pth"))

criterion = torch.nn.L1Loss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# TEST
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10, anneal_strategy="linear"
# )
lmbda = lambda epoch: 0.65 ** epoch
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
evaluator = PCQM4MEvaluator()

train(net, criterion, optimizer, 10, scheduler=scheduler)
eval(net, evaluator)

# 0.784012496471405 without edges after 1 epoch
# 0.783663272857666 with edges (not embedded) after 1 epoch
# 0.8013435006141663 with edges (embedded) after 1 epoch
