import torch
import matplotlib.pyplot as plt
from torch_geometric.nn.conv import MessagePassing
from model_utils import F_loss_fn
from data_get_utils import get_mini_dataloader

def mini_train(model: MessagePassing, rho: float, num_items: int, batch_size: int, num_epochs: int) -> None:
    """trains model on dataloader, saves weights of the best-performing model, and logs ongoing results through wandb.
          
    parameters
    ----------
    model : MessagePassing
        self-explanatory.
    rho : float
        loss = (1-rho)*E_loss + rho*F_loss.
    num_epochs : int
        self-explanatory.
    batch_size : int
        self-explanatory.
    num_epochs: int
        self-explanatory.
    """
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

    train_dataloader = get_mini_dataloader(molecule='benzene',
            num_items=num_items,
            batch_size=batch_size)

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
            
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()