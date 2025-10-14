import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataLoader, TensorDataset, random_split,  Subset
from sklearn.model_selection import train_test_split

import Types

sequenceMapping = {
    "N": torch.tensor([0,0,0,0], dtype=torch.float32),
    "A": torch.tensor([0,0,0,1], dtype=torch.float32),
    "C": torch.tensor([0,0,1,0], dtype=torch.float32),
    "G": torch.tensor([0,1,0,0], dtype=torch.float32),
    "T": torch.tensor([1,0,0,0], dtype=torch.float32)
}

########## ----------- Dataset Helper Functions --------- ##########

def toDataloaders(
        dataFrame: pd.core.frame.DataFrame,
        validationSplit: float,
        testSplit: float,
        trainBatchSize: int,
        valBatchSize: int,
        testBatchSize: int,
        seed: int
    ):

    """
    Function that takes a DataFrame (sequence and label columns) as argument,\n
    Calculates onehot encoded sequences and masks for onehot, combines all together and\n
    shuffles the dataset then splits it into training, validation and testing DataLoaders.

    Parameters
    ----------
    dataFrame : DataFrame
        DataFrame to split and create 3 DataLoaders.

    trainSplit : float
        Size out of 1 of training's DataLoader.
        
    validationSplit : float
        Size out of 1 of validation's DataLoader.

    testSplit : float
        Size out of 1 of testing's DataLoader.

    trainBatchSize : int
        Batch size of training's DataLoader.

    valBatchSize : int
        Batch size of validation's DataLoader.
    
    testBatchSize : int
        Batch size of testing's DataLoader.
    
    seed : int
        Seed for shuffle.
    
    Returns
    ----------
    DataLoader
        train, validation, test
    """

    dataFrame["sequence"] = dataFrame["sequence"].astype(str)
    dataFrame["label"] = dataFrame["label"].astype(int)

    tempDataFrame, testDataFrame = train_test_split(
        dataFrame,
        test_size=testSplit,
        random_state=seed,
        stratify=dataFrame["label"],
    )

    validationSplit = validationSplit / (1 - testSplit)
    trainDataFrame, validationDataFrame = train_test_split(
        tempDataFrame,
        test_size=validationSplit,
        random_state=seed,
        stratify=tempDataFrame["label"],
    )

    def build_items(df):
        items = []
        for seq, lbl in zip(df["sequence"].tolist(), df["label"].tolist()):
            x = sequenceTo1Hot(seq).to(torch.float32)
            x = x.permute(1, 0).contiguous()
            mask = (x.sum(dim=0) > 0).to(torch.float32)
            y = torch.tensor(lbl, dtype=torch.long)
            items.append((x, mask, seq, y))
        return items

    trainData = build_items(trainDataFrame)
    validationData = build_items(validationDataFrame)
    testData = build_items(testDataFrame)

    train = DataLoader(
        trainData,
        batch_size=trainBatchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    validation = DataLoader(
        validationData,
        batch_size=valBatchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test = DataLoader(
        testData,
        batch_size=testBatchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train, validation, test

def printDataloader(name: str, data: DataLoader) -> None:
    """
    Prints DataLoader's name, number of batches, samples and shapes from onehot encoded and mask onehot.

    Parameters
    ----------
    name : str
        Name of the DataLoader (Training/Validation/Testing).

    data : DataLoader
    """

    batches = len(data)
    samples = len(data.dataset)

    onehot, maskonehot, sequences, y = next(iter(data))

    Helpers.colourPrint(Types.Colours.PURPLE, f"Print stats for {name} DataLoader and 3 rows")
    Helpers.colourPrint(Types.Colours.PURPLE, f"  - Number of batches: {batches}")
    Helpers.colourPrint(Types.Colours.PURPLE, f"  - Number of samples: {samples}")
    Helpers.colourPrint(Types.Colours.PURPLE, f"  - Onehot encoded sequences shape: {onehot.shape}")
    Helpers.colourPrint(Types.Colours.PURPLE, f"  - Mask for onehot encoded sequences shape: {maskonehot.shape}")
    Helpers.colourPrint(Types.Colours.PURPLE, f"  - Sequences batch size: {len(sequences)}")
    Helpers.colourPrint(Types.Colours.PURPLE, f"  - Labels shape: {len(y)}")

    try:
        batch = next(iter(data))
        onehot, maskonehot, sequence, y = batch

        table = []

        for i in range(3):
            
            table.append({
                "onehot": onehot[i, :, :10].flatten().tolist()[:10],
                "mask": maskonehot[i].flatten().tolist()[:10],
                "index": i,
                "sequences": sequence[i],
                "labels": y[i]
            })

        df = pd.DataFrame(table)
        print(df)

    except StopIteration:
        raise ValueError("Empty DataLoader was given")

def loadDatasetPercentage(datasetPath: str, percentage: int)-> DatasetDict:
    
    """
    Shuffle the dataset, take a percentage, and split into training and validation sets.

    Parameters
    ----------
    datasetPath : str
        Relative path to the CSV dataset.

    percentage : float
        Percentage of the dataset to use.

    Returns
    -------
    dict
        Dictionary containing the training and validation splits.
    """

    fullDataset = load_dataset("csv",data_files=datasetPath,split="train")
    fullDataset = fullDataset.shuffle(seed=Types.DEFAULT_SMORFCNN_SEED)
    k = int(len(fullDataset) * percentage / 100)
    fullDataset = fullDataset.select(range(k))

    split = fullDataset.train_test_split(
        test_size=Types.DEFAULT_DNABER6_TEST_SPLIT,
        seed=Types.DEFAULT_SMORFCNN_SEED,
    )

    return split

########## ----------- End --------- ##########

########## ----------- Sequence Helper Functions --------- ##########

def kmer(sequence: str, k: int):

    """
    Function that splits a DNA sequence into k-mer parts.

    Parameters
    ----------
    sequence : str
        DNA sequence as a string.

    k : int
        Size of each k-mer.

    Returns
    ----------
    list
        string items of size k.
    """

    kmerSequence = []

    for i in range(len(sequence) - k + 1):
        
        kmer = sequence[i:i+k]
        kmerSequence.append(kmer)

    return kmerSequence

def kmerDnabert(sequence: str, k: int, ambiguousState: Types.KmerAmbiguousState):

    """
    Function that splits a DNA sequence into k-mer parts.

    Parameters
    ----------
    sequence : str
        DNA sequence as a string.

    k : int
        Size of each k-mer.

    ambiguousState : Enum
        Masking to use for unknown characters for DNABERT.

    Returns
    ----------
    list
        string items of size k.
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
    Helper function to create 1-hot encoded tensor, from a DNA sequence input, padded to 512 default length.

    Parameters
    ----------
    sequence : str
        DNA sequence.

    Returns
    ----------
    Tensor
        DNA sequence as 1hot encoded Tensor.
    """

    encoded = torch.zeros(Types.DEFAULT_DNABERT6_WINDOW_SIZE,4)

    for index, nt in enumerate(sequence):

        if nt not in sequenceMapping:
            raise ValueError(f"Unexpected character “{nt}” in DNA sequence!")        

        encoded[index] = sequenceMapping.get(nt)

    return encoded

########## ----------- End --------- ##########

########## ----------- Pooling Functions --------- ##########

def globalMaxPooling(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    """
    Calculate `Masked Global Max`. For 1hot encoded sequences,\n
    make padded positions to -inf using the mask so they cant win the MAX.

    Parameters
    ----------
    x : Tensor
        input Tensor.
    mask : Tensor
        mask of input Tensor, shape [B, 1, L].

    Returns
    ----------
    Tensor
        Global Max Tensor, of shape [B, C].
    """

    m = mask[:, None, :]

    xNInf = x.masked_fill(m == 0, float("-inf"))
    globalMax = xNInf.max(dim=-1).values
    globalMax = torch.where(torch.isfinite(globalMax), globalMax, torch.zeros_like(globalMax))

    return globalMax

def globalAveragePooling(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    """
    Calculate `Masked Global Average`. For 1hot encoded sequences,\n
    zero-out padded positions using the mask so they are not included to sum.

    Parameters
    ----------
    x : Tensor
        input as Tensor

    mask : Tensor
        mask of input.

    Returns
    ----------
    Tensor
        Returns Global Average, of shape [B, C].
    """

    m = mask[:, None, :]
    mFloat = m.to(dtype=x.dtype)

    xZero = x * mFloat
    denom = mFloat.sum(dim=-1).clamp(min=1.0)
    globalAverage = xZero.sum(dim=-1) / denom 

    return globalAverage

########## ----------- End --------- ##########

########## ----------- Compute Metrics Functions --------- ##########

def computeEpochMetrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    runningLoss: float,
    n: int,
    threshold: float,
    epochIndex: int
) -> dict:
    """
    Compute epoch-level metrics and convert runningLoss to sample epoch loss.

    Parameters
    ----------
    probabillities : Tensor
        Calculated from model's sigmoid.

    targets : Tensor
        Labels for each item from dataset.

    runningLoss : float
        Epoch's current loss.

    n : int
        size of dataset.

    threshold : float
        Base float number for positive-negative.

    epochIndex : int
        Current epoch.

    Returns
    ----------
    dict
        {'loss','acc','precision','recall','f1','TP','TN','FP','FN'}.
    """

    probabilities = probabilities.view(-1)
    targets = targets.view(-1).to(dtype=torch.long)

    yPred = (probabilities >= threshold)
    positive = (targets == 1)
    negative = (targets == 0)

    truePositive = (yPred & positive).sum().item()
    falsePositive = (yPred & negative).sum().item()
    falseNegative = ((~yPred) & positive).sum().item()
    trueNegative = ((~yPred) & negative).sum().item()

    total = truePositive + trueNegative + falsePositive + falseNegative

    accuraccy = (truePositive + trueNegative) / total
    precision = truePositive / (truePositive + falsePositive)
    recall = truePositive / (truePositive + falseNegative)
    f1 = (2 * precision * recall) / (precision + recall)

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
    """
    Computes Epoch AUC, True Positive Ration and False Positive Ration.

    Parameters
    ----------
    probabillities : Tensor
        Calculated from model's sigmoid.

    targets : Tensor
        labels for each item from dataset.

    epochIndex : int
        Current epoch.

    Returns
    ----------
    dict
        {"auc", "tpr", "fpr"}.
    """

    probabilities = probabilities.detach().view(-1).cpu().numpy().astype(np.float64)
    targets = targets.detach().view(-1).cpu().numpy().astype(np.int32)

    positives = (targets == 1).sum()
    negatives = (targets == 0).sum()

    order = np.argsort(-probabilities)
    ySorted = targets[order]

    truePositive = np.cumsum(ySorted == 1)
    falsePositive = np.cumsum(ySorted == 0)
    truePositiveRatio = truePositive / positives
    falsePositiveRatio = falsePositive / negatives

    falsePositiveRatio = np.concatenate([[0.0], falsePositiveRatio, [1.0]])
    truePositiveRatio = np.concatenate([[0.0], truePositiveRatio, [1.0]])
    auc = float(np.trapz(truePositiveRatio, falsePositiveRatio))
    
    aucDict = {"auc": auc, "fpr": falsePositiveRatio, "tpr": truePositiveRatio}

    printEpochAUC(aucDict, epochIndex)

    return aucDict

def kFoldSummary(foldMetrics: list) -> dict:
    """
    Calculates summary from kFold Cross Validation.\n
    Mean and Std for keys ---> "loss", "acc", "precision", "recall", "f1", "auc", "learningRate".\n
    Total sum for keys    ---> "TP", "TN", "FP", "FN".

    Parameters
    ----------
    foldMetrics : list
        Contains all metrics k, calculated in KFoldCrossValidation.

    Returns
    ----------
    dict
        with mean/std for scalar metrics and sums for counts.
    """

    meanKeys = ["loss", "acc", "precision", "recall", "f1", "auc", "learningRate"]
    sumKeys = ["TP", "TN", "FP", "FN"]

    summary = {}

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

########## ----------- End --------- ##########

########## ----------- Print Metrics Function --------- ##########

def printEpochMetrics(metrics: dict, epochIndex: int) -> None:
    """
    Prints epoch metrics as DataFrame.

    Parameters
    ----------
    metrics : dict
        Contains per Epoch metrics.

    epochIndex : int
        Current epoch.
    """

    df = pd.DataFrame([metrics])
    print(f"Epoch {epochIndex} Metrics and TP, FP, TN, FN")
    print(df.to_string(index=False))

def printEpochAUC(auc: dict, epochIndex: int) -> None:
    """
    Prints epoch AUC.

    Parameters
    ----------
    auc : dict
        Contains for per Epoch auc metrics.

    epochIndex : int
        Current epoch.
    """

    auc = {k: v for k, v in auc.items() if k not in ("fpr", "tpr")}
    print(f"Epoch {epochIndex} AUC: {auc["auc"]}")

def printFitSummary(trainingMetrics: dict, validationMetrics: dict) -> None:
    """
    Prints summary of fit (train-validation-test) as DataFrame.

    Parameters
    ----------
    trainingMetrics : dict
        Contans epoch-size lists for each metric key in training.

    validationMetrics : dict
        Contans epoch-size lists for each metric key in validation.
    """
    tMetricsDf = pd.DataFrame(trainingMetrics)
    vMetricsDf = pd.DataFrame(validationMetrics)

    tMetricsDf.insert(0, "epoch", np.arange(1, len(tMetricsDf) + 1, dtype=int))
    vMetricsDf.insert(0, "epoch", np.arange(1, len(vMetricsDf) + 1, dtype=int))

    vMetricsDf = vMetricsDf.drop(columns=["fpr", "tpr"], errors="ignore")

    print("Summary of Metrics computed during Fit")
    print(tMetricsDf.to_string(index=False))
    print(vMetricsDf.to_string(index=False))

def printKFoldMetrics(
    foldMetrics: list,
    foldMetricsSummary: dict
)-> None:
    """
    Prints fold metrics and metrics summary as DataFrame.

    Parameters
    ----------
    foldMetrics : list
        k-size list containing every fold's metrics.

    foldMetricsSummary : dict
        Contains summary (sum,mean and std) of fold metrics.
    """

    df = pd.DataFrame(foldMetrics)
    print("\nPer-fold validation metrics:")
    print(df)

    print("\n10-Fold summary (mean ± std and sums):")
    df = pd.DataFrame(foldMetricsSummary)
    print(df)

########## ----------- End --------- ##########

########## ----------- Plot Functions --------- ##########

def plotLabelDistribution(
        trainDataLoader: DataLoader,
        validationDataLoader: DataLoader,
        testDataLoader: DataLoader
    ):
    """
    Plots bar diagram for label distirbution in each dataset (train-val-test)
    and bar diagram for the total dataset.

    Parameters
    ----------
    trainDataLoader : DataLoader
        Data for model training.

    validationDataLoader : DataLoader
        Data for model validation.

    testDataLoader : DataLoader
        Data for model testing.
    """

    totalLength = len(trainDataLoader.dataset)
    totalLength += len(validationDataLoader.dataset)
    totalLength += len(testDataLoader.dataset)

    totalPositiveTrain = totalPositiveVal = totalPositiveTest = 0
    totalNegativeTrain = totalNegativeVal = totalNegativeTest = 0

    for _, _, _, y in trainDataLoader:
        y = y.detach().cpu()
        totalPositiveTrain += int((y == 1).sum().item())
        totalNegativeTrain += int((y == 0).sum().item())
    
    for _, _, _, y in validationDataLoader:
        y = y.detach().cpu()
        totalPositiveVal += int((y == 1).sum().item())
        totalNegativeVal += int((y == 0).sum().item())    

    for _, _, _, y in testDataLoader:
        y = y.detach().cpu()
        totalPositiveTest += int((y == 1).sum().item())
        totalNegativeTest += int((y == 0).sum().item())  

    totalPositive = totalPositiveTrain + totalPositiveVal + totalPositiveTest

    splits = ["Training", "Validation", "Test"]
    negatives = [totalNegativeTrain, totalNegativeVal, totalNegativeTest]
    positives = [totalPositiveTrain, totalPositiveVal, totalPositiveTest]

    x = np.arange(len(splits), dtype=float)
    width = 0.38
    gap = 0.08

    fig1 = plt.figure()
    plt.bar(x - (width/2 + gap/2), negatives, width=width, label="Label 0 (Non-coding)")
    plt.bar(x + (width/2 + gap/2), positives, width=width, label="Label 1 (Coding)")
    plt.xticks(x, splits)
    plt.ylabel("count")
    plt.title("Label distribution per split")
    plt.legend()
    plt.tight_layout()

    totalNegative = int(totalLength - totalPositive)

    fig2 = plt.figure()
    plt.bar(["Label 0 (Non-Coding)", "Label 1 (Coding)"], [totalNegative, totalPositive], width=0.4)
    plt.ylabel("count")
    plt.title("Label distribution (all splits combined)")
    plt.tight_layout()

    plt.show()

    return fig1, fig2

def plotConfusionPie(
    trainingMetrics: list,
    validationMetrics: list,
    testMetrics: dict,
    epochs: int
):
    """
    Plot Pie chart for True Positive, True Negative, False Positve and False Negative,
    computed during fit of model (train-validation-test).

    Parameters
    ----------
    trainingMetrics : dict 
        Contains epoch-size lists for each metric key in training.

    validationMetrics : dict
        Contains epoch-size lists for each metric key in validation.

    testMetrics : dict
        Contains epoch-size lists for each metric key in testing.

    epochs : int
        Total epochs of fit.
    """

    TP,TN,FP,FN,total = 0,0,0,0,0

    for epoch in range(0, epochs):
        TP += trainingMetrics["TP"][epoch] + validationMetrics["TP"][epoch]
        TN += trainingMetrics["TN"][epoch] + validationMetrics["TN"][epoch]
        FP += trainingMetrics["FP"][epoch] + validationMetrics["FP"][epoch]
        FN += trainingMetrics["FN"][epoch] + validationMetrics["FN"][epoch]

    TP += testMetrics["TP"]
    TN += testMetrics["TN"]
    FP += testMetrics["FP"]
    FN += testMetrics["FN"]

    total = TP + TN + FP + FN

    fig = plt.figure()
    sizes = [TP, TN, FP, FN]
    labels = ["TP", "TN", "FP", "FN"]

    plt.pie(sizes, labels=labels, autopct=lambda p: f"{p:.1f}%")
    plt.title("Confusion proportions on Traing-Validation-Testing")
    plt.tight_layout()

    plt.show()

    return fig

def plotFitCurves(
    trainingMetrics: dict,
    validationMetrics: dict
):
    """
    Creates 3 separate figures: Loss, Accuracy, F1. Comparing training with validation.

    Parameters
    ----------
    trainingMetrics : dict
    Contans epoch-size lists for each metric key in training.

    validationMetrics : dict
    Contans epoch-size lists for each metric key in validation.
    """

    epochs = len(trainingMetrics["loss"])
    x = np.arange(1, epochs + 1)

    fig1 = plt.figure()
    plt.plot(x, trainingMetrics["loss"], label="Training Loss")
    plt.plot(x, validationMetrics["loss"], label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend()

    fig2 = plt.figure()
    plt.plot(x, trainingMetrics["acc"], label="Training Accuracy")
    plt.plot(x, validationMetrics["acc"], label="Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy"); plt.legend()

    fig3 = plt.figure()
    plt.plot(x, trainingMetrics["f1"], label="Training F1")
    plt.plot(x, validationMetrics["f1"], label="Validation F1")
    plt.xlabel("Epoch"); plt.ylabel("F1"); plt.title("F1"); plt.legend()

    plt.show()

    return fig1, fig2, fig3

def plotROCCurve(validationMetrics: dict, epoch: int):
    """
    Plot ROC AUC Curve for best epoch during fit.

    Parameters
    ----------
    validationMetrics : dict
        Contans epoch-size lists for each metric key in validation.

    epoch : int
        Best Epoch.
    """

    if epoch == 0 or epoch is None:
        raise ValueError("Epoch must have an integer value > 0")

    fpr = np.asarray(validationMetrics.get("fpr", []), dtype=float)[epoch - 1]
    tpr = np.asarray(validationMetrics.get("tpr", []), dtype=float)[epoch - 1]
    auc = float(validationMetrics.get("auc")[epoch - 1])

    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend()

    plt.show()

    return fig

def plotMeanROC(
    foldMetrics: list,
    summary: dict
):
    """
    Plot ROC on FPR grid and mean ROC.

    Parameters
    ----------
    foldMetrics : list
        Contains every fold's metrics.

    summary : dict
        Contains summary (sum,mean and std) of fold metrics.
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

    plt.show()

    return fig

########## ----------- End --------- ##########

def colourPrint(colour: Types.Colours, msg: str) -> None:

    PRINT_BLUE = "\033[34m"
    PRINT_GREEN = "\033[32m"
    PRINT_RED = "\033[31m"
    PRINT_PURPLE = "\033[95m
    PRINT_RESET = "\033[0m"

    if colour == Types.Colours.WHITE:
        print(f"{msg}")

    elif colour == Types.Colours.BLUE:
        print(f"{PRINT_BLUE}{msg}{PRINT_RESET}")
    
    elif colour == Types.Colours.GREEN:
        print(f"{PRINT_GREEN}{msg}{PRINT_RESET}")

    elif colour == Types.Colours.RED:
        print(f"{PRINT_RED}{msg}{PRINT_RESET}")

    elif colour == Types.Colours.PURPLE:
        print(f"{PRINT_PURPLE}{msg}{PRINT_RESET}")
