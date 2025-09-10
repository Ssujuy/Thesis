import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataLoader, TensorDataset, random_split
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
    truePositiveRatio = truePositive / positives
    falsePositiveRatio = falsePositive / negatives
    # add (0,0) and (1,1)
    falsePositiveRatio = np.concatenate([[0.0], falsePositiveRatio, [1.0]])
    truePositiveRatio = np.concatenate([[0.0], truePositiveRatio, [1.0]])
    auc = float(np.trapz(truePositiveRatio, falsePositiveRatio))
    
    aucDict = {"auc": auc, "fpr": falsePositiveRatio, "tpr": truePositiveRatio}

    printEpochAUC(aucDict, epochIndex)

    return aucDict

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
            value = float(fold.get(key, 0.0))
            values.append(value)

        summary[f"{key}Mean"] = float(np.nanmean(np.array(values, dtype=float)))
        summary[f"{key}Std"]  = float(np.nanstd(np.array(values, dtype=float)))

    for key in sumKeys:
        total = 0
        for fold in foldMetrics:
            value = fold.get(key, 0)
            total += int(value)

        summary[f"Total{key}"] = int(total)

    return summary

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

def plotLabelDistribution(
        trainDataLoader: DataLoader,
        validationDataLoader: DataLoader,
        testDataLoader: DataLoader
    ):
    
    totalLength = len(trainDataLoader.dataset)
    totalLength += len(validationDataLoader.dataset)
    totalLength += len(testDataLoader.dataset)

    yTrain = trainDataLoader.dataset["y"].view(-1).cpu()
    yVal = validationDataLoader.dataset["y"].view(-1).cpu()
    yTest = testDataLoader.dataset["y"].view(-1).cpu()

    totalPositiveTrain = int((yTrain == 1).sum().item())
    totalPositiveVal = int((yVal == 1).sum().item())
    totalPositiveTest = int((yTest == 1).sum().item())

    totalNegativeTrain = int((yTrain == 0).sum().item())
    totalNegativeVal = int((yVal == 0).sum().item())
    totalNegativeTest = int((yTest == 0).sum().item())

    totalPositive = totalPositiveTrain + totalPositiveVal + totalPositiveTest

    # ---- 1) Grouped bar: label 0 vs 1 per split ----
    splits = ["Training", "Validation", "Test"]
    negatives = [totalNegativeTrain, totalNegativeVal, totalNegativeTest]
    positives = [totalPositiveTrain, totalPositiveVal, totalPositiveTest]

    x = np.arange(len(splits), dtype=float)
    width = 0.4

    fig1 = plt.figure()
    plt.bar(x - width/2, negatives, width=width, label="Label 0 (Non-coding)")
    plt.bar(x + width/2, positives, width=width, label="Label 1 (Coding)")
    plt.xticks(x, splits)
    plt.ylabel("count")
    plt.title("Label distribution per split")
    plt.legend()
    plt.tight_layout()

    # ---- 2) Overall bar: all splits combined ----
    totalNegative = int(totalLength - totalPositive)

    fig2 = plt.figure()
    plt.bar(["Label 0 (Nojn-Coding)", "Label 1 (Coding)"], [totalNegative, totalPositive])
    plt.ylabel("count")
    plt.title("Label distribution (all splits combined)")
    plt.tight_layout()

def plotConfusionPie():
    return

def plotFitCurves(
    trainingMetrics: dict,
    validationMetrics: dict
):
    """
    history: dict from fit() with keys:
      train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, (optional) lr
    Creates 3 separate figures: Loss, Accuracy, F1.
    """
    epochs = len(trainingMetrics["loss"])

    # Loss
    fig1 = plt.figure()
    plt.plot(epochs, trainingMetrics["loss"], label="Training Loss")
    plt.plot(epochs, validationMetrics["loss"],   label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend()

    # Accuracy
    fig2 = plt.figure()
    plt.plot(epochs, trainingMetrics["acc"], label="Training Accuracy")
    plt.plot(epochs, validationMetrics["acc"], label="Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy"); plt.legend()

    # F1
    fig3 = plt.figure()
    plt.plot(epochs, trainingMetrics["f1"], label="Training F1")
    plt.plot(epochs, validationMetrics["f1"], label="Validation F1")
    plt.xlabel("Epoch"); plt.ylabel("F1"); plt.title("F1"); plt.legend()

    return fig1, fig2, fig3

def plotROCCurve(validationMetrics: dict, epoch: int):
    """
    metrics must contain: 'fpr' (list), 'tpr' (list), 'auc' (float) or 'roc_auc'.
    """

    if epoch == 0 or epoch is None:
        raise ValueError("Epoch must have an integer value > 0")

    fpr = np.asarray(validationMetrics[epoch - 1].get("fpr", []), dtype=float)
    tpr = np.asarray(validationMetrics[epoch - 1].get("tpr", []), dtype=float)
    auc = float(validationMetrics[epoch - 1].get("auc"))

    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend()
    return fig


def plotMeanROC(
    foldMetrics: list,
    summary: dict
):
    """
    Interpolate each fold's ROC onto a common FPR grid, plot mean ROC with ±1 SD band.
    If `summary` is provided (from kFoldSummary), its aucMean/aucStd are used for the legend.
    """
    grid = np.linspace(0.0, 1.0, 1001)
    tprs = []
    aucs = []

    for fold in foldMetrics:
        fpr = np.asarray(fold["fpr"], dtype=float)
        tpr = np.asarray(fold["tpr"], dtype=float)

        tprs.append(np.interp(grid, fpr, tpr))
        aucs.append(float(fold.get("auc",np.nan)))

    tprs = np.vstack(tprs)
    meanTpr = np.nanmean(tprs, axis=0)
    stdTpr  = np.nanstd(tprs, axis=0)

    aucMean = float(summary.get("aucMean", np.nanmean(aucs)))
    aucStd  = float(summary.get("aucStd",  np.nanstd(aucs)))


    fig = plt.figure()
    plt.plot(grid, meanTpr, label=f"mean AUC = {aucMean:.3f} ± {aucStd:.3f}")
    lower = np.clip(meanTpr - stdTpr, 0.0, 1.0)
    upper = np.clip(meanTpr + stdTpr, 0.0, 1.0)
    plt.fill_between(grid, lower, upper, alpha=0.2)
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Mean ROC K-fold Cross Validation"); plt.legend()
    return fig