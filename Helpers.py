import torch

sequenceMapping = {
    "N": torch.tensor([0,0,0,0], dtype=torch.float32),
    "A": torch.tensor([0,0,0,1], dtype=torch.float32),
    "C": torch.tensor([0,0,1,0], dtype=torch.float32),
    "G": torch.tensor([0,1,0,0], dtype=torch.float32),
    "T": torch.tensor([1,0,0,0], dtype=torch.float32)
}

def sequenceTo1Hot(sequence: str)-> torch.Tensor:
    """
    Helper function to create 1-hot encoded tensor
    from a DNA sequence input
    """
    L = len(sequence)
    encoded = torch.zeros(L,4)
    
    for index, nt in enumerate(sequence):
        
        if nt not in sequenceMapping:
            raise ValueError(f"Unexpected character “{nt}” in DNA sequence!")        
        
        encoded[index] = sequenceMapping.get(nt)
        
    return encoded