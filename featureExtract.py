import argparse, dnabert6, Types, Helpers


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
    projectionState: Types.ProjectionState,
    projectionDimension: int,
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
        projectionState=projectionState,
        projectionDimension=projectionDimension,
        hiddenState=hiddenState,
        saveDir=saveDir
    )

    model.datasetInit()
    model.finetune()
    print("Run 'features' Mode to extract embeddings from DNABERT6 and other features for CNN Classifier!")

def featureExtraction(
    modelDirectory: str,
    sequencesPath: str,
    saveFeaturesPath: str,
    projectionState: Types.ProjectionState,
    projectionDimension: int,
    hiddenState: Types.HiddenState,
):
    
    model = dnabert6.DNABERT6(projectionState=projectionState, projectionDimension=projectionDimension, hiddenState=hiddenState)
    model.load(modelDirectory)

    sequence = ""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fine tune DNABERT6 model for coding vs non-coding smORFs with 'finetune' Mode." \
        "Or load DNABERT6 model to extract embeddings and compute other features with 'features' Mode."
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    ft = sub.add_parser("finetune", help="Finetune DNABERT6 model")
    ft.add_argument("--datasetPath", type=str, required=True, help="CSV with 'sequence' + 'label' columns.")
    ft.add_argument("--datasetPercentage", type=int, default=100, help="N% of the dataset that will be used for finetune.")
    ft.add_argument("--epochs", type=int, default=Types.DEFAULT_DNABERT6_EPOCHS)
    ft.add_argument("--learningRate", type=float, default=Types.DEFAULT_DNABERT6_LEARNING_RATE)
    ft.add_argument("--windowSize", type=int, default=Types.DEFAULT_DNABERT6_WINDOW_SIZE, help="Maximum length of each 6mer sequence.")
    ft.add_argument("--weightDecay", type=float, default=Types.DEFAULT_DNABERT6_WEIGHT_DECAY)
    ft.add_argument("--warmupRatio", type=float, default=Types.DEFAULT_DNABERT6_WARMUP_RATIO)
    ft.add_argument("--finetuneEvalBatchSize", type=int, default=Types.DEFAULT_DNABERT6_BATCH_SIZE)
    ft.add_argument("--finetuneTrainBatchSize", type=int, default=Types.DEFAULT_DNABERT6_BATCH_SIZE)
    ft.add_argument("--embeddingsBatchSize", type=int, default=Types.DEFAULT_DNABERT6_BATCH_SIZE)
    ft.add_argument("--projectionState", type=int, choices=[0, 1, 2], default=0, help="0=no projection, 1=frozen (non-trainable) projection, 2=trainable projection.")
    ft.add_argument("--projectionDimension", type=int, default=None, help="Size of the embeddings projection.")
    ft.add_argument("--hiddenState", type=str, choices=["cls", "mean", "both"], default='cls', help="Pooling: 'cls', 'mean', or 'both'.")
    ft.add_argument("--saveDirectory", type=str, default=Types.DEFAULT_DNABER6_SAVE_DIRECTORY,help="Directory to save the finetuned model.")

    emb = sub.add_parser("features", help="Load finetuned model and extract embeddings from dna sequences.")
    emb.add_argument("--modelDirectory", type=str, required=True, default=Types.DEFAULT_DNABER6_SAVE_DIRECTORY)
    emb.add_argument("--sequencesPath", type=str, required=True)
    emb.add_argument("--saveFeaturesPath", type=str, required=True)
    emb.add_argument("--projectionState", type=int, choices=[0, 1, 2], default=0, help="0=no projection, 1=frozen (non-trainable) projection, 2=trainable projection.")
    emb.add_argument("--projectionDimension", type=int, default=None, help="Size of the embeddings projection.")
    emb.add_argument("--hiddenState", type=str, choices=["cls", "mean", "both"], default='cls', help="Pooling: 'cls', 'mean', or 'both'.")

    args = parser.parse_args()

    if args.cmd == "finetune":

        print("Initializing DNABERT6 model and starting finetune process")

        projStateMap = {
            0: Types.ProjectionState.NO_PROJECTION,
            1: Types.ProjectionState.NOT_TRAINABLE,
            2: Types.ProjectionState.TRAINABLE,
        }
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
            projStateMap[args.projectionState],
            args.projectionDimension,
            hiddenStateMap[args.hiddenState],
            args.saveDirectory
        )

    elif args.cmd == "features":

        print("Loading DNABERT6 model to extract embeddings and compute other features")

        projectionState = None
        hiddenState = None

        if args.projectionState == 0:
            projectionState = Types.ProjectionState.NO_PROJECTION
        elif args.projectionState == 1:
            projectionState = Types.ProjectionState.NOT_TRAINABLE
        elif args.projectionState == 2:
            projectionState = Types.ProjectionState.TRAINABLE
        else:
            raise ValueError("Projection State accepts values 0,1,2")

        if args.hiddenState == 'cls':
            hiddenState = Types.HiddenState.CLS
        elif args.hiddenState == 'mean':
            hiddenState = Types.HiddenState.MEAN
        elif args.hiddenState == 'both':
            hiddenState = Types.HiddenState.BOTH
        else:
            raise ValueError("Hidden State accepts values 'cls', 'mean' or 'both'")

        featureExtraction(
            args.modelDirectory,
            args.sequencesPath,
            args.saveFeaturesPath,
            projectionState,
            args.projectionDimension,
            hiddenState,
        )
    
    else:
        raise ValueError("Only 'features' or 'finetune' Modes are supported")