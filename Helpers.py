import torch
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import Dataloader, TensorDataset, random_split
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

def toDataloaders(
        dataset: TensorDataset,
        trainSplit: float,
        validationSplit: float,
        testSplit: float,
        trainBatchSize: int,
        valBatchSize: int,
        testBatchSize: int,
        seed: int
    ):

    """
    Function that takes a TensorDataset as argument, shuffles the dataset
    then splits it into training, validation and testing DataLoaders.
    """

    n = len(dataset)
    gen = torch.Generator().manual_seed(seed)
    trainDs, validationDs, testDs = random_split(dataset, [trainSplit * n, validationSplit * n, testSplit * n], generator=gen)
    
    train = DataLoader(
        trainDs,
        batch_size=trainBatchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    validation = DataLoader(
        validationDs,
        batch_size=valBatchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test = DataLoader(
        testDs,
        batch_size=testBatchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train, validation, test

def printDataloader(name: str, dataL: Dataloader) -> None:

    print(f"Print 3 rows for {name} DataLoader with shape: {dataL.shape}")

    try:
        batch = next(iter(dataL))
        xOnehot, maskOnehot, xEmbed, maskEmbed, y = batch

        table = []

        for i in range(3):
            
            table.append({
                "index": i,
                "onehot": xOnehot[i],
                "maskOnehot": maskOnehot[i],
                "embeddings": xEmbed[i],
                "maskEmbedings": maskEmbed[i],
                "labels": y[i]
            })

        df = pd.DataFrame(table)
        print(df)

    except StopIteration:
        raise ValueError("Empty DataLoader was given")

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

def computeEpochMetrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    runningLoss: float,
    n: int,
    threshold: float,
    epochIndex: int
) -> dict:
    """
    Compute epoch-level metrics and convert runningLoss to sample epoch loss.\n
    Returns: {'loss','acc','precision','recall','f1','TP','TN','FP','FN'}\n
    """

    # Flatten & types
    probabilities = probabilities.view(-1)
    targets = targets.view(-1).to(dtype=torch.long)

    # Threshold → predictions
    yPred = (probabilities >= threshold)
    positive = (targets == 1)
    negative = (targets == 0)

    #Calculating TP, TN, FP, FN
    truePositive = (yPred & positive).sum().item()
    falsePositive = (yPred & negative).sum().item()
    falseNegative = ((~yPred) & positive).sum().item()
    trueNegative = ((~yPred) & negative).sum().item()

    total = truePositive + trueNegative + falsePositive + falseNegative

    accuraccy = (truePositive + trueNegative) / total
    precision = truePositive / truePositive + falsePositive
    recall = truePositive / truePositive + falseNegative
    f1 = 2 * precision * recall / precision + recall

    epochLoss = runningLoss / n

    metrics = {
        "loss": epochLoss,
        "acc": accuraccy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": truePositive,
        "TN": trueNegative,
        "FP": falsePositive,
        "FN": falseNegative,
    }

    printEpochMetrics(metrics, epochIndex)

    return metrics

def computeEpochROC(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    epochIndex: int
) -> dict:

    probabilities = probabilities.detach().cpu().numpy().astype(np.float64)
    targets = targets.detach().cpu().numpy().astype(np.int32)
    # sort by descending score
    order = np.argsort(-probabilities)
    sorted = targets[order]
    positives = (sorted == 1).sum()
    negatives = (sorted == 0).sum()

    truePositive = int(np.cumsum(sorted == 1))
    falsePositive = int(np.cumsum(sorted == 0))
    truePositive = truePositive / positives
    falsePositiveRatio = falsePositive / negatives
    # add (0,0) and (1,1)
    falsePositiveRatio = np.concatenate([[0.0], falsePositiveRatio, [1.0]])
    truePositive = np.concatenate([[0.0], truePositive, [1.0]])
    auc = float(np.trapz(truePositive, falsePositiveRatio))
    
    aucDict = {"auc": auc, "fpr": falsePositiveRatio, "tpr": truePositive}

    printEpochAUC(aucDict, epochIndex)

    return aucDict

def computePR(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    epochIndex: int
)-> dict:
    
    probabilities = probabilities.detach().cpu().view(-1).numpy()
    targets = targets.detach().cpu().view(-1).numpy().astype(np.int32)

    positives = (targets == 1).sum()
    negatives = (targets == 0).sum()

    #     # Raw precision/recall at each distinct score step
    # precision = tp / np.maximum(tp + fp, 1.0)
    # recall    = tp / P

    # # Keep thresholds aligned with raw points (useful for plotting/inspection)
    # thresholds = s

    # # ---- interpolate precision (monotone envelope) ----
    # # Make precision non-increasing w.r.t recall (right-to-left max)
    # # This matches the "interpolated precision" used in AP definitions.
    # precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]

    # # ---- step-wise AP (VOC/sklearn-style) ----
    # # Insert sentinel endpoints
    # mrec = np.concatenate(([0.0], recall, [1.0]))
    # mpre = np.concatenate(([1.0], precision_envelope, [0.0]))
    # # Enforce envelope again across sentinels (safety)
    # for i in range(mpre.size - 1, 0, -1):
    #     mpre[i-1] = max(mpre[i-1], mpre[i])
    # # Sum area of horizontal steps where recall increases
    # idx = np.where(mrec[1:] != mrec[:-1])[0]
    # ap = float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))

def printEpochMetrics(metrics: dict, epochIndex: int) -> None:
    df = pd.DataFrame(metrics)
    print(f"Epoch {epochIndex} Metrics and TP, FP, TN, FN")
    print(df)

def printEpochAUC(auc: dict, epochIndex: int) -> None:
    df = pd.DataFrame(auc)
    print(f"Epoch {epochIndex} AUC , False Positive Ratio and False Negative Ratio")
    print(df)

def printFitSummary(
        trainingMetrics: dict,
        validationMetrics: dict
    ) -> None:
        
        tMetricsDf  =  pd.DataFrame({
            trainingMetrics
        })

        vMetricsDf = pd.DataFrame({
            validationMetrics
        })

        print("----------########## Summary of Metrics computed during Fit ##########----------")
        print(tMetricsDf)
        print(vMetricsDf)

def kFoldSummary(foldMetrics: list) -> dict:
    """
    fold_metrics_list: list of dicts, each with keys:
      loss, acc, precision, recall, learningRate, f1, TP, TN, FP, FN, auc, fpr, tpr
    Returns a 'summary' dict with mean/std for scalar metrics and sums for counts.
    """
    meanKeys = ["loss", "acc", "precision", "recall", "f1", "auc", "learningRate"]
    sumKeys = ["TP", "TN", "FP", "FN"]

    summary = {}

    # --- mean/std over folds for scalar metrics ---
    for key in meanKeys:
        values = []
        for fold in foldMetrics:
            value = float(v)
            values.append(v)

        summary[f"{key}Mean"] = float(np.nanmean(np.array(values, dtype=float)))
        summary[f"{key}Std"]  = float(np.nanstd(np.array(values, dtype=float)))

    for key in sumKeys:
        total = 0
        for fold in foldMetrics:
            value = fold.get(key, 0)
            total += int(value)

        summary[f"Total{key}"] = int(total)

    return summary

def printKFoldMetrics(
    foldMetrics: list,
    foldMetricsSummary: dict
)-> None:
    

    df = pd.DataFrame(foldMetrics)
    print("\nPer-fold validation metrics:")
    print(df)

    print("\n10-Fold summary (mean ± std and sums):")
    df = pd.DataFrame(foldMetricsSummary)
    print(df)