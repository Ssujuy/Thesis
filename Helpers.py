import torch
from pathlib import Path
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import TensorDataset, random_split
import Types

sequenceMapping = {
    "N": torch.tensor([0,0,0,0], dtype=torch.float32),
    "A": torch.tensor([0,0,0,1], dtype=torch.float32),
    "C": torch.tensor([0,0,1,0], dtype=torch.float32),
    "G": torch.tensor([0,1,0,0], dtype=torch.float32),
    "T": torch.tensor([1,0,0,0], dtype=torch.float32)
}

########## ----------- PyTorch File Helper Functions --------- ##########


def saveFeaturesPtFile(
        saveFeaturesPath: str,
        sequences: list,
        onehot: torch.tensor,
        embeddings: torch.tensor,
        labels: torch.tensor,
        metadata: dict
) -> None:

    """
    Function that takes args sequences, onehot, embeddings, labels and metadata
    and saves them to PyTorch file, for training of smORF CNN classifier.
    """

    # Save everything in one .pt file
    payload = {
        "sequences": sequences,
        "onehot": onehot,
        "embeddings": embeddings,
        "labels": labels,
        "metadata": metadata,
    }
    saveFeaturesPath = Path(saveFeaturesPath)
    torch.save(payload, saveFeaturesPath)

    print(f"✓ Saved {len(sequences)} samples to {saveFeaturesPath.resolve()}")
    print(f"   onehot:      {tuple(onehot.shape)}")
    print(f"   embeddings:  {tuple(embeddings.shape)}")
    print(f"   labels:      {tuple(labels.shape)}")


def printPt(
        saveFeaturesPath: str,
        rows:int =Types.DEFAULT_PT_ROWS_PRINT,
        dim: int =Types.DEFAULT_PT_LENGTH_PRINT
    ):

    """
    Prints a features pt file's first 6 (default) rows with max length 10. 
    """

    saveFeaturesPath = Path(saveFeaturesPath)
    ptFile = torch.load(saveFeaturesPath, map_location="cpu")

    print(f"Printing first {rows} rows of {saveFeaturesPath.resolve()} for sanity check.")

    oh = ptFile["onehot"][:rows, :dim, :].to(torch.int8).cpu().tolist()
    emb = ptFile["embeddings"][:rows, :dim].cpu().numpy().tolist()

    df = pd.DataFrame({
        "sequence": ptFile["sequences"][:rows],
        "label": ptFile["labels"][:rows].tolist(),
        "onehot": oh,
        "embeddings": emb
    })

    print(df)

########## ----------- End --------- ##########

########## ----------- Dataset Helper Functions --------- ##########


def loadFeaturesFromPt(path: str) -> TensorDataset:

    """
    Loads features.pt and returns TensorDataset:
      (x_onehot [N,4,L], mask_onehot [N,L], x_embed [N,D,1], mask_embed [N,1], y [N])
    """

    features = torch.load(Path(path), map_location="cpu")

    if features["embeddings"] == None and features["onehot"] == None:
        raise ValueError(f"Pytorch file {path} does not contain key 'onehot' and embeddings")

    onehot = torch.as_tensor(features["onehot"], dtype=torch.float32)   # [N,L,4]
    xOnehot = onehot.permute(0, 2, 1).contiguous()              # [N,4,L]
    maskOnehot = (onehot.sum(dim=-1) > 0).to(torch.float32)     # [N,L]

    # --- embeddings: numpy [N,D] -> tensor [N,D,1], mask = ones ---
    embnp = features["embeddings"]                                     # (N,D) numpy
    xEmbed = torch.from_numpy(embnp).to(torch.float32).unsqueeze(-1)  # [N,D,1]
    maskEmbed = torch.ones((xEmbed.size(0), 1), dtype=torch.float32)  # [N,1]


    if features["labels"] != None:
        # --- labels ---
        y = torch.as_tensor(features["labels"], dtype=torch.long)           # [N]
    
    else:
        raise ValueError(f"Pytorch file {path} does not contain key 'labels'")

    return TensorDataset(xOnehot, maskOnehot, xEmbed, maskEmbed, y)

def shuffleSPlitTDataset(
        dataset: TensorDataset,
        trainSplit: float,
        validationSplit: float,
        testSplit: float,
        seed: int
    ):

    """
    Function that takes a TensorDataset as argument, shuffles the dataset
    then splits it into training, validation and testing.
    """

    n = len(dataset)
    gen = torch.Generator().manual_seed(seed)
    train, validation, test = random_split(dataset, [trainSplit * n, validationSplit * n, testSplit * n], generator=gen)
    return train, validation, test


def loadDatasetPercentage(datasetPath: str, percentage: int)-> DatasetDict:
    
    """
    Function that shuffles, takes a percentage of the dataset
    then splits for training and validation.
    """

    fullDataset = load_dataset("csv",data_files=datasetPath,split="train")
    fullDataset = fullDataset.shuffle(seed=42)
    k = int(len(fullDataset) * percentage / 100)
    fullDataset = fullDataset.select(range(k))

    split = fullDataset.train_test_split(
        test_size=0.2,
        seed=42,
    )

    return split

########## ----------- End --------- ##########

########## ----------- Sequence Helper Functions --------- ##########

def kmer(sequence: str, k: int, ambiguousState: Types.KmerAmbiguousState):

    """
    Function that splits a DNA sequence into k-mer parts.
    """

    kmerSequence = []

    for i in range(len(sequence) - k + 1):
        
        kmer = sequence[i:i+k]

        if 'N' in kmer:

            if ambiguousState == Types.KmerAmbiguousState.MASK:
                kmerSequence.append('[MASK]')
            else:
                kmerSequence.append('[UNK]')
        else:
            kmerSequence.append(kmer)

    return kmerSequence

def sequenceTo1Hot(sequence: str)-> torch.Tensor:
    """
    Helper function to create 1-hot encoded tensor
    from a DNA sequence input, padded to 512 default length.
    """
    encoded = torch.zeros(Types.DEFAULT_DNABERT6_WINDOW_SIZE,4)
    
    for index, nt in enumerate(sequence):
        
        if nt not in sequenceMapping:
            raise ValueError(f"Unexpected character “{nt}” in DNA sequence!")        
        
        encoded[index] = sequenceMapping.get(nt)
        
    return encoded

########## ----------- End --------- ##########

########## ----------- Pooling Functions --------- ##########

def globalAveragePooling(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    """
    Calculate `Masked Global Max`. For 1hot encoded sequences
    make padded positions to -inf using the mask so they cant win the MAX.

    mask --> [B, 1, L]
    globalMax --> [B, C]
    """

    # broadcast mask over channels
    m = mask[:, None, :]

    xNInf = x.masked_fill(m == 0, float("-inf"))
    globalMax = xNInf.max(dim=-1).values
    globalMax = torch.where(torch.isfinite(globalMax), globalMax, torch.zeros_like(globalMax))

    return globalMax

def globalMaxPooling(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    """
    Calculate `Masked Global Average`. For 1hot encoded sequences
    zero-out padded positions using the mask so they are not included to sum.

    denom --> [B, 1]
    globalAverage --> [B, C]
    """

    m = mask[:, None, :]
    mFloat = m.to(dtype=x.dtype)

    xZero = x * mFloat
    denom = mFloat.sum(dim=-1).clamp(min=1.0)
    globalAverage = xZero.sum(dim=-1) / denom 

    return globalAverage

########## ----------- End --------- ##########