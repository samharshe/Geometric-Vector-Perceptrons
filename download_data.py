from typing import Any
import torch
from torch_geometric.datasets import MD17
import torch_geometric.transforms as T
from torch_geometric.transforms import RadiusGraph, NormalizeScale

# minumum energy in benzene, uracil, and aspirin datasets is -406757.5938
max_abs_energy = -406757.5938

class NormalizeForce(T.BaseTransform):
    def __call__(self, data: Any) -> Any:
        data.force = torch.div(data.force, max_abs_energy)
        
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class NormalizeEnergy(T.BaseTransform):
    def __call__(self, data: Any) -> Any:
        data.energy = torch.div(data.energy, max_abs_energy)
        
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class DoubleDistance(T.BaseTransform):
    def __call__(self, data: Any) -> Any:
        data.pos = data.pos*2
        
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

transform = T.Compose([RadiusGraph(1.8100), NormalizeScale(), DoubleDistance(), NormalizeEnergy(), NormalizeForce()])

benzene_dataset = MD17(root='data/', name='benzene', pre_transform=transform)