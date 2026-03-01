import pandas as pd
import numpy as np, torch
from pathlib import Path
import Types, Helpers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    log_loss,
    precision_recall_fscore_support,
)

class DNABERT6:
    """
    Class DNABERT6 constructs the pre-trained DNABERT-6 model (Jhi & Zhou, 2021), along with essential parameters for the model's fine-tune or feature extraction.
    DNABERT6 accepts 6-mer overlapping input of DNA sequences and produces embeddings.
    DNABERT6 class has the option of loading a fine-tuned model and draw embeddings from a list of sequences or be fine-tuned (so the model can adjust on specific data) and saved.

    Attributes
    ----------
    trainingDataPath : str
        File path to csv Dataset for model fine-tune.
    
    trainDatasetPercentage : int    
        Percentage of the fine-tune Dataset to be used (0-100).

    epochs : int
        Epochs of fine-tuning.

    learningRate : float
        Learning Rate for fine-tuning.

    windowSize : int
        Maximum window size for DNA sequences.

    weightDecay : float
        Weight decay for fine-tune.

    warmupRatio : float
        Warmpup ratio for fine-tune.

    fineTuneEvalBatchSize : int
        Batch size for fine-tuning evaluation.

    fineTuneTrainBatchSize : int
        Batch size for fine-tune training.

    embeddingsBatchSize : int
        Batch size for embeddings.
    
    device: str
        Device to move model and parameters.
    
    hiddenState : Types.HiddenState
        Type of embeddings to be returned, (CLS, MEAN, BOTH).
        CLS: return embeddings from the CLS token.
        MEAN: return embeddings from mean average of all tokens, except CLS, SEP, PAD.
        BOTH: return concatenation of CLS + MEAN.

    strategy : str
        Strategy for fine-tuning.

    metric : str
        Metric to calcualte model inprovment in fine-tuning.

    saveDir : str
        Directory path to save the model.

    Methods
    ----------
    datasetInit() -> None
        Initialize dataset csv file, tokenise every row with DNABERT's tokenizer.
        Finally, create pyTorch tensor and assign training and validation Datasets to member variables.

    _poolHidden(self, hidden, attentionMask, state: Types.HiddenState, specialTokensMask=None)
        With hidden(B, L, 768), last_hidden_state from DNABERT and attentionMask for valid positions,
        embeddings are returned for a batch of sequences.

        For state CLS: embeddings from [CSL] token are returned, size: (B, 768).
        For state MEAN: mean average of embeddings from all tokens except [CLS], [SEP] and [PAD] are returned, size: (B, 768).
        For state BOTH: CLS and MEAN embeddings are concatenated, size: (B,1536). 
    
    metrics(self, evalPred) -> dict
        Function given to Trainer to evaluate metrics: (accuracy, f1_score, loss, precision).
        Returns a dictionary {"accuracy", "precision", "recall", "f1", "cross_entropy"}.
    
    encode(self, batch) -> AutoTokenizer
        Convert batched string to sequences to kmers and return tokenizer.
        Return an AutoTokenizer with batchedKmers.

    finetune(self, **override) -> None
        Fine-tune the DNABERT-6 model with coding and non-coding labeled smORFs taken from our train.csv.
        Initialize TrainingArguments and Trainer, fine-tune then save model and training arguments to directory.

    embeddings(self, sequences: list) -> torch.Tensor
        Calculate DNABERT-6 embeddings for a list of sequences. Return a pyTorch Tensor with size (B,768).
        for hidden state  CLS or MEAN or a size of (B, 1536) for hidden state BOTH.

    load(self, modelPath: str) -> None:
        Fully restore a fine-tuned checkpoint from modelPath directory.
    
    history(self) -> pd.DataFrame
        Return log_history as DataFrame.

    parameters(self) -> None
        Print model's parameters.
    """
    def __init__(
                self,
                trainingDataPath: str                   = Types.DEFAULT_DNABERT6_DATASET_PATH,
                trainDatasetPercentage: int             = Types.DEFAULT_DNABERT6_DATASET_PERCENTAGE,
                epochs: int                             = Types.DEFAULT_DNABERT6_EPOCHS,
                learningRate: float                     = Types.DEFAULT_DNABERT6_LEARNING_RATE,
                windowSize: int                         = Types.DEFAULT_DNABERT6_WINDOW_SIZE,
                weightDecay: float                      = Types.DEFAULT_DNABERT6_WEIGHT_DECAY,
                warmupRatio: float                      = Types.DEFAULT_DNABERT6_WARMUP_RATIO,
                fineTuneEvalBatchSize: int              = Types.DEFAULT_DNABERT6_BATCH_SIZE,
                fineTuneTrainBatchSize: int             = Types.DEFAULT_DNABERT6_BATCH_SIZE,
                embeddingsBatchSize: int                = Types.DEFAULT_DNABERT6_BATCH_SIZE,
                device: str                             = Types.DEFAULT_DNABERT6_DEVICE,
                hiddenState: Types.HiddenState          = Types.HiddenState.BOTH,
                strategy: str                           = Types.DEFAULT_DNABERT6_STRATEGY,
                metric: str                             = Types.DEFAULT_DNABERT6_METRIC,
                saveDir : str                           = Types.DEFAULT_DNABERT6_SAVE_DIRECTORY
            ):
        """
        Constructor for DNABERT6 class. Initializes member variables and DNABERT-6 pre-trained model, that can be fine-tuned or used to draw embeddings from a list of sequences.

        Attributes
        ----------
        trainingDataPath : str
            File path to csv Dataset for model fine-tune.
        
        trainDatasetPercentage : int    
            Percentage of the fine-tune Dataset to be used (0-100).

        epochs : int
            Epochs of fine-tuning.

        learningRate : float
            Learning Rate for fine-tuning.

        windowSize : int
            Maximum window size for DNA sequences.

        weightDecay : float
            Weight decay for fine-tune.

        warmupRatio : float
            Warmpup ratio for fine-tune.

        fineTuneEvalBatchSize : int
            Batch size for fine-tuning evaluation.

        fineTuneTrainBatchSize : int
            Batch size for fine-tune training.

        embeddingsBatchSize : int
            Batch size for embeddings.
        
        device: str
            Device to move model and parameters.
        
        hiddenState : Types.HiddenState
            Type of embeddings to be returned, (CLS, MEAN, BOTH).
            CLS: return embeddings from the CLS token.
            MEAN: return embeddings from mean average of all tokens, except CLS, SEP, PAD.
            BOTH: return concatenation of CLS + MEAN.

        strategy : str
            Strategy for fine-tuning.

        metric : str
            Metric to calcualte model inprovment in fine-tuning.

        saveDir : str
            Directory path to save the model.
        """

        self.modelID = Types.DEFAULT_DNABERT6_MODEL_ID
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelID)
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(self.modelID, num_labels=2)
        self.hiddenState = hiddenState

        if self.hiddenState == Types.HiddenState.BOTH:
            self.outputSize = 2 * Types.DEFAULT_DNABERT6_OUTPUT_SIZE

        else:
            self.outputSize = Types.DEFAULT_DNABERT6_OUTPUT_SIZE

        self.trainingDatasetPercentage = trainDatasetPercentage
        self.trainingDataPath = trainingDataPath
        self.trainDataset = None
        self.validationDataset = None
        self.windowSize = windowSize
        self.learningRate = learningRate
        self.epochs = epochs
        self.weightDecay = weightDecay
        self.warmupRatio = warmupRatio
        self.fineTuneTrainBatchSize = fineTuneTrainBatchSize
        self.fineTuneEvalBatchSize = fineTuneEvalBatchSize
        self.embeddingsBatchSize = embeddingsBatchSize
        self.strategy = strategy
        self.metric = metric
        self.saveDirectory = saveDir

        self.trainer = None
        self.arguments = None

        Helpers.colourPrint(Types.Colours.BLUE, "Initialized DNABERT-6 model")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Dataset path for finetuning: {self.trainingDataPath}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Dataset percentage to use: {self.trainingDatasetPercentage}%")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Learning rate: {self.learningRate}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Window size: {self.windowSize}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Weight decay: {self.weightDecay}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Warmup ratio: {self.warmupRatio}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Finetuning eval barch size: {self.fineTuneEvalBatchSize}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Finetuning train batch size: {self.fineTuneTrainBatchSize}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Embeddings batch size: {self.embeddingsBatchSize}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Output size: {self.outputSize}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Hidden state: {self.hiddenState}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Evaluation, logging and save stratefy: {self.strategy}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Metric for best model: {self.metric}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Directory to save the finetuned model: {self.saveDirectory}")

    def datasetInit(self) -> None:
        """
        Initialize dataset csv file, tokenise every row with DNABERT's tokenizer.
        Finally, create pyTorch tensor and assign training and validation Datasets to member variables.
        """

        split = Helpers.loadDatasetPercentage(self.trainingDataPath, self.trainingDatasetPercentage)

        self.trainDataset = split["train"]
        self.validationDataset = split["test"]

        self.trainDataset = self.trainDataset.map(self.encode, batched=True, remove_columns=["sequence"])
        self.validationDataset = self.validationDataset.map(self.encode, batched=True, remove_columns=["sequence"])

        self.trainDataset.set_format(type="torch")
        self.validationDataset.set_format(type="torch")

        Helpers.colourPrint(Types.Colours.PURPLE, f"Initialized Training Dataset: {self.trainDataset.shape}")
        Helpers.colourPrint(Types.Colours.PURPLE, f"Initialized Validation Dataset: {self.validationDataset.shape}")

    def _poolHidden(self, hidden, attentionMask, state: Types.HiddenState, specialTokensMask=None) -> torch.Tensor:
        """
        With hidden(B, L, 768), last_hidden_state from DNABERT and attentionMask for valid positions embeddings are returned for a batch of sequences.

        For state CLS: embeddings from [CSL] token are returned, size: (B, 768).
        For state MEAN: mean average of embeddings from all tokens except [CLS], [SEP] and [PAD] are returned, size: (B, 768).
        For state BOTH: CLS and MEAN embeddings are concatenated, size: (B,1536).

        Parameters
        ----------
        hidden
            DNABERT's last hidden state.

        attentionMask
            Mask for padded positions of the sequence.

        state : Types.HiddenState
            Which embeddings to return CLS, MEAN or BOTH.

        specialTokensMask
            Mask for special tokens [CLS], [SEP], etc.

        Return
        ----------
        Tensor
            DNABERT-6 embeddings Tensor.
        """

        if state == Types.HiddenState.CLS:

            return hidden[:, 0, :]
        
        elif state == Types.HiddenState.MEAN:

            if specialTokensMask is not None:
                valid = (attentionMask.bool() & (~specialTokensMask.bool()))
            else:
                valid = attentionMask.bool()
                valid[:, 0] = False

            mask = valid.to(dtype=hidden.dtype).unsqueeze(-1)

            summed  = (hidden * mask).sum(dim=1)
            counts  = mask.sum(dim=1).clamp_min(1e-9)
            mean    = summed / counts
            return mean

        elif state == Types.HiddenState.BOTH:

            clsEmd = self._poolHidden(hidden, attentionMask, Types.HiddenState.CLS)
            meanEmb = self._poolHidden(hidden, attentionMask, Types.HiddenState.MEAN)

            combined = torch.cat([clsEmd, meanEmb], dim=-1)
            return combined

        else:
            raise ValueError("state must be CLS, MEAN or BOTH")

    def metrics(self, evalPred: tuple) -> dict:
        """
        Function given to Trainer to evaluate metrics: (accuracy, f1_score, loss, precision).

        Parameters
        ----------        
        evalPred : tuple
            Evalutation prediction from DNABERT6 Trainer.
        
        Return
        ----------
        dict
            Dictionary of metrics {"accuracy", "precision", "recall", "f1", "cross_entropy"}.
        """

        logits = (
            evalPred.predictions[0]
            if isinstance(evalPred.predictions, tuple)
            else evalPred.predictions
        )

        labels = evalPred.label_ids

        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels)

        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        preds = probs.argmax(axis=1)

        acc  = accuracy_score(labels, preds)
        f1   = f1_score(labels, preds)
        prec, rec, _, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

        ce_loss = log_loss(labels, probs,labels=[0, 1])

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "cross_entropy": ce_loss,
        } 

    def encode(self, batch) -> AutoTokenizer:
        """
        Convert batched string to sequences to kmers and return tokenizer

        Parameters
        ----------
        batch : list
            Batch of sequences.
        
        Return
        ----------
        AutoTokenizer
            Tokenizer for DNABERT-6 model with batchedKmers.
        """

        batchKmers = [Helpers.kmerDnabert(seq, Types.DEFAULT_DNABERT6_KMER_SIZE, Types.KmerAmbiguousState.MASK) for seq in batch["sequence"]]

        return self.tokenizer(
            batchKmers,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.windowSize,
            return_special_tokens_mask=True
        )

    def finetune(self, **override) -> None:
        """
        Fine-tune the DNABERT-6 model with coding and non-coding labeled smORFs taken from our train.csv.
        Initialize TrainingArguments and Trainer, fine-tune then save model and training arguments to directory.
        """

        self.args = TrainingArguments(
            output_dir                  = self.saveDirectory,
            num_train_epochs            = self.epochs,
            per_device_train_batch_size = self.fineTuneTrainBatchSize,
            per_device_eval_batch_size  = self.fineTuneEvalBatchSize,
            learning_rate               = self.learningRate,
            weight_decay                = self.weightDecay,
            warmup_ratio                = self.warmupRatio,
            eval_strategy               = self.strategy,
            save_strategy               = self.strategy,
            logging_strategy            = self.strategy,
            load_best_model_at_end      = True,
            metric_for_best_model       = self.metric,
            greater_is_better           = True,
            max_grad_norm               = 1.0,
        )

        self.trainer = Trainer(
            self.model,
            self.args,
            train_dataset               = self.trainDataset,
            eval_dataset                = self.validationDataset,
            compute_metrics             = self.metrics,
        )

        self.trainer.train()
        self.trainer.save_model(self.saveDirectory)
        self.tokenizer.save_pretrained(self.saveDirectory)
        Helpers.colourPrint(Types.Colours.GREEN, f"DNABERT-6 fine-tuned model saved to  {Path(self.saveDirectory).resolve()}")

    def embeddings(self, sequences: list) -> torch.Tensor:
        """
        Calculate DNABERT-6 embeddings for a list of sequences. Return a pyTorch Tensor with size (B,768) for hidden state  CLS or MEAN or a size of (B, 1536) for hidden state BOTH.

        Parameters
        ----------
        sequences : list
            List of sequencers to calculate embeddings.
        
        Return
        ----------
        Tensor
            Embeddings Tensor for sequences list.
        """

        self.model.eval()
        self.model.to(self.device)

        out = torch.empty((len(sequences), self.outputSize), device=self.device, dtype=torch.float32)
        idx = 0

        with torch.no_grad():

            for i in range(0, len(sequences), self.embeddingsBatchSize):

                batch = sequences[i : i + self.embeddingsBatchSize]
                batchedKmers = [Helpers.kmerDnabert(seq, Types.DEFAULT_DNABERT6_KMER_SIZE, Types.KmerAmbiguousState.MASK) for seq in batch]

                toks  = self.tokenizer(
                    batchedKmers,
                    is_split_into_words=True,
                    truncation=True,
                    padding="max_length",
                    max_length=self.windowSize,
                    return_tensors="pt",
                    return_special_tokens_mask=True
                ).to(self.device)

                attentionMask = toks["attention_mask"]
                specialTokensMask  = toks["special_tokens_mask"]
                                
                hidden = self.model.base_model(input_ids=toks["input_ids"], attention_mask=attentionMask).last_hidden_state
                pooled = self._poolHidden(hidden, attentionMask, self.hiddenState, specialTokensMask)

                bsz = pooled.size(0)
                out[idx : idx + bsz] = pooled.to(dtype=torch.float32)
                idx += bsz

        return out

    def load(self, modelPath: str) -> None:
        """
        Fully restore a fine-tuned checkpoint from modelPath directory.

        Parameters
        ----------
        modelPath : str
            Path to model's save directory.
        """
        path = Path(modelPath)

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()

        argsPath = path/"training_args.bin"
        if argsPath.exists():
            self.args = torch.load(argsPath, weights_only=False)

            self.epochs = int(self.args.num_train_epochs)
            self.learningRate = float(self.args.learning_rate)
            self.weightDecay = float(self.args.weight_decay)
            self.warmupRatio = float(self.args.warmup_ratio)

        else:
            self.args = None

        Helpers.colourPrint(Types.Colours.GREEN, f"Loaded DNABERT-6 fine-tuned model from: {path.resolve()}")

    def history(self) -> pd.DataFrame:
        """
        Return log_history as DataFrame.

        Return
        ----------
        DataFrame
            Log history as DataFrame.
        """
        history = self.trainer.state.log_history
        return pd.DataFrame(history)

    def parameters(self) -> None:
        """
        Print model's parameters.
        """
        Helpers.colourPrint(Types.Colours.BLUE, f"{self.model.named_parameters()}")