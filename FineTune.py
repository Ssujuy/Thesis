import argparse, dnabert6, csv
import Types, Helpers
import pandas as pd
from pathlib import Path
import torch

def finetune(
    datasetPath: str,
    datasetPercentage: str,
    epochs: int,
    learningRate: int,
    windowSize: int,
    weightDecay: float,
    warmupRatio: float,
    finetuneEvalBatchSize: int,
    finetuneTrainBatchSize: int,
    embeddingsBatchSize: int,
    hiddenState: Types.HiddenState,
    saveDir: str
):

    model = dnabert6.DNABERT6(
        trainingDataPath=datasetPath,
        trainDatasetPercentage=datasetPercentage,
        epochs=epochs,
        learningRate=learningRate,
        windowSize=windowSize,
        weightDecay=weightDecay,
        warmupRatio=warmupRatio,
        fineTuneEvalBatchSize=finetuneEvalBatchSize,
        fineTuneTrainBatchSize=finetuneTrainBatchSize,
        embeddingsBatchSize=embeddingsBatchSize,
        hiddenState=hiddenState,
        saveDir=saveDir
    )

    model.datasetInit()
    model.finetune()
    print("Run 'features' Mode to extract embeddings from DNABERT6 and other features for CNN Classifier!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fine tune DNABERT6 model for coding vs non-coding smORFs with 'finetune' Mode."
    )

    parser.add_argument("--datasetPath", type=str, required=True, help="CSV with 'sequence' + 'label' columns.")
    parser.add_argument("--datasetPercentage", type=int, default=100, help="N% of the dataset that will be used for finetune.")
    parser.add_argument("--epochs", type=int, default=Types.DEFAULT_DNABERT6_EPOCHS)
    parser.add_argument("--learningRate", type=float, default=Types.DEFAULT_DNABERT6_LEARNING_RATE)
    parser.add_argument("--windowSize", type=int, default=Types.DEFAULT_DNABERT6_WINDOW_SIZE, help="Maximum length of each 6mer sequence.")
    parser.add_argument("--weightDecay", type=float, default=Types.DEFAULT_DNABERT6_WEIGHT_DECAY)
    parser.add_argument("--warmupRatio", type=float, default=Types.DEFAULT_DNABERT6_WARMUP_RATIO)
    parser.add_argument("--finetuneEvalBatchSize", type=int, default=Types.DEFAULT_DNABERT6_BATCH_SIZE)
    parser.add_argument("--finetuneTrainBatchSize", type=int, default=Types.DEFAULT_DNABERT6_BATCH_SIZE)
    parser.add_argument("--embeddingsBatchSize", type=int, default=Types.DEFAULT_DNABERT6_BATCH_SIZE)
    parser.add_argument("--hiddenState", type=str, choices=["cls", "mean", "both"], default='both', help="Pooling: 'cls', 'mean', or 'both'.")
    parser.add_argument("--saveDirectory", type=str, default=Types.DEFAULT_DNABER6_SAVE_DIRECTORY,help="Directory to save the finetuned model.")

    args = parser.parse_args()

    print("Initializing DNABERT6 model and starting finetune process")

    hiddenStateMap = {
        "cls":  Types.HiddenState.CLS,
        "mean": Types.HiddenState.MEAN,
        "both": Types.HiddenState.BOTH,
    }

    finetune(
        args.datasetPath,
        args.datasetPercentage,
        args.epochs,
        args.learningRate,
        args.windowSize,
        args.weightDecay,
        args.warmupRatio,
        args.finetuneEvalBatchSize,
        args.finetuneTrainBatchSize,
        args.embeddingsBatchSize,
        hiddenStateMap[args.hiddenState],
        args.saveDirectory
    )