import torch, sys, wandb
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.loader import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Callable, Dict
from model_utils import F_loss_fn

def train(model: MessagePassing, optimizer: Optimizer, loss_fn: Callable, train_dataloader: DataLoader, rho: float, num_epochs: int) -> [float]:
    """trains model on dataloader, saves weights of the best-performing model, and logs ongoing results through wandb.
          
    parameters
    ----------
    model : MessagePassing
        self-explanatory.
    optimizer : Optimizer
        self-explanatory.
    loss_fn : Callable
        self-explanatory.
    train_dataloader : DataLoader
        self-explanatory.
    rho : float
        loss = (1-rho)*E_loss + rho*F_loss.
    num_epochs : int
        self-explanatory.
    """
    losses = []
    # training loop occurs num_epochs times
    for _ in range(num_epochs):
        # track gradients
        model.train()
        
        # iterate through test_dataloader        
        for data in train_dataloader:
            
            # clear gradients
            optimizer.zero_grad()

            # target values
            E = data.energy
            F = data.force
            
            # predictions from the model
            E_hat, F_hat = model(data)
            E_hat.squeeze_(dim=1)

            # squared error for energy loss
            E_loss = loss_fn(E_hat, E)

            # a version of squared error for force loss
            F_loss = F_loss_fn(F_hat, F, loss_fn)
            
            # canonical loss
            loss = (1-rho)*E_loss + rho*F_loss
        
            # calculate gradients
            loss.backward()
            
            # keep track of losses
            losses.append(loss.item())
            
            # update
            optimizer.step()
        
    return losses