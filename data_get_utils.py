from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch.utils.data import random_split
    
def get_mini_dataloader(molecule: str, num_items: int, batch_size: int) -> DataLoader:
    """returns a DataLoader object as specified in function call; especially useful for getting small DataLoader objects to use in experimentation.
    
    parameters
    ----------
    molecule : str
        which of molecule datasets (benzene, uracil, aspirin) to fetch.
    num_items : int
        self-explanatory. 
    batch_size : int
        self-explanatory.
        
    returns
    -------
    DataLoader object as specified in function call.
    """
    # load in the dataset
    dataset = MD17(root='data/', name=f'{molecule}', pre_transform=None, force_reload=False)

    # make mini_dataset out of dataset
    mini_dataset, _ = random_split(dataset, [num_items, len(dataset)-num_items])
    
    # make min_dataloader out of mini_dataset
    mini_dataloader = DataLoader(mini_dataset, batch_size=batch_size)

    # return DataLoader
    return mini_dataloader

def get_molecule(type: str) -> Data:
    """returns the rirst item in dataset of type of molecule specified in function call.
    
    parameters
    ----------
    molecule : str
        molecule whose first instance in dataset will be returned.
        
    returns
    -------
    first data item in dataset of molecule specified in function call.
    """
    return MD17(root='data/', name=f'{type}', pre_transform=None, force_reload=False)[0]
