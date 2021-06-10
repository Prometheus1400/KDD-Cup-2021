# conda activate my-rdkit-env
from ogb.lsc import PygPCQM4MDataset, PCQM4MEvaluator
from ogb.utils import smiles2graph
from torch_geometric.data import DataLoader
import torch
from tqdm import tqdm
from gin import Net

import os
import numpy as np


def get_freer_gpu():
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_mem_file")
    memory_available = [
        int(x.split()[2]) for x in open("tmp_mem_file", "r").readlines()
    ]
    os.system("rm tmp_mem_file")
    return np.argmax(memory_available)


device = torch.device(f"cuda:{get_freer_gpu()}")

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
batch_size = 256
train_loader = DataLoader(
    dataset[split_idx["train"]], batch_size=batch_size, shuffle=True
)
valid_loader = DataLoader(
    dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    dataset[split_idx["test"]], batch_size=batch_size, shuffle=False
)


def train(model, loss, optimizer, epochs, save=True, scheduler=None):
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        prev_loss = 1000.0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}: Loss NAN: Slope NAN: LR NAN ",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
        for i, batch in enumerate(pbar):
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
            if i % 200 == 199:  # print every 2000 mini-batches
                # print(f"[{epoch+1}], [{i+1}] loss: {running_loss/2000}")
                loss = round(running_loss / 200, 3)
                # each slope "step" is 2000 iterations
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_description(
                    f"Epoch {epoch}: Loss {loss}: Slope {round((loss - prev_loss),4)}: LR {lr} "
                )
                prev_loss = loss
                running_loss = 0.0

        if scheduler != None and epoch != 0 and epoch % 30 == 0:
            scheduler.step()

    print("Finished Training")
    if save:
        torch.save(net.state_dict(), "GIN/Saves/GIN_100_epochs_ReduceLRonPlateue.pth")
        print("Saved")


def eval(model, evaluator):
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for data in tqdm(valid_loader, desc="Evalutating"):
            data = data.to(device)
            outputs = model(data).view(-1)
            y_true.append(data.y.view(outputs.shape).detach().cpu())
            y_pred.append(outputs.detach().cpu())

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        input_dict = {"y_true": y_true, "y_pred": y_pred}
        result_dict = evaluator.eval(input_dict)

    # Result
    print(result_dict["mae"])


net = Net(
    num_tasks=1,
    num_layers=5,
    emb_dim=300,
    gnn_type="gin",
    drop_ratio=0,
    graph_pooling="sum",
    JK="last",
    residual=False,
)
net.to(device)
# net.load_state_dict(torch.load("GIN/Saves/GIN_100_epochs_initBatchNorm.pth"))

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# lmbda = lambda epoch: 0.25 ** epoch
# scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.25,
    patience=10,
    threshold=0.0001,
    threshold_mode="rel",
    cooldown=0,
    min_lr=0,
    eps=1e-08,
    verbose=True,
)
evaluator = PCQM4MEvaluator()

train(net, criterion, optimizer, 100, scheduler=scheduler, save=True)
eval(net, evaluator)
