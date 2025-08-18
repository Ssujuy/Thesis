import pandas as pd
import numpy as np, torch
from pathlib import Path
import Types, Helpers
import torch.nn.functional as F
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

### The projection to fixed size needs to be tested with real data coding, non coding

### read fasta file for prediction. 
class DNABERT6:

    def __init__(
                self,
                modelID: str                            = Types.DEFAULT_DNABERT6_MODEL_ID,
                trainingDataPath: str                   = Types.DEFAULT_DNABER6_DATASET_PATH,
                trainDatasetPercentage                  = Types.DEFAULT_DNABER6_DATASET_PERCENTAGE,
                epochs: int                             = Types.DEFAULT_DNABERT6_EPOCHS,
                learningRate: float                     = Types.DEFAULT_DNABERT6_LEARNING_RATE,
                windowSize: int                         = Types.DEFAULT_DNABERT6_WINDOW_SIZE,
                weightDecay: float                      = Types.DEFAULT_DNABERT6_WEIGHT_DECAY,
                warmupRatio: float                      = Types.DEFAULT_DNABERT6_WARMUP_RATIO,
                fineTuneEvalBatchSize                   = Types.DEFAULT_DNABERT6_BATCH_SIZE,
                fineTuneTrainBatchSize                  = Types.DEFAULT_DNABERT6_BATCH_SIZE,
                embeddingsBatchSize                     = Types.DEFAULT_DNABERT6_BATCH_SIZE,
                device: str                             = Types.DEFAULT_DNABERT6_DEVICE,
                projectionState: Types.ProjectionState  = Types.ProjectionState.NO_PROJECTION,
                projectionDimension: int                = None,
                hiddenState: Types.HiddenState          = Types.HiddenState.CLS,
                saveDir : str                           = Types.DEFAULT_DNABER6_SAVE_DIRECTORY
            ):

        # tokenizer and model (with classification head)

        self.tokenizer = AutoTokenizer.from_pretrained(modelID)

        # For finding coding - non-coding smORFs we need classification, thus
        # nothing is frozen, backbone + classifier head all have requires_grad=True

        # Device cuda or cpu

        self.device = device

        self.model = AutoModelForSequenceClassification.from_pretrained(modelID, num_labels=2)

        # Initialize hidden state (CLS,MEAN,BOTH)

        self.hiddenState = hiddenState

        # Inititalize projection state, projection dimension and projection member variables

        self.projectionState = projectionState
        self.linear = None

        if self.hiddenState == Types.HiddenState.BOTH:
            self.projectionDimension = 2 * Types.DEFAULT_DNABERT6_PROJECTION_DIMENSION

        else:
            self.projectionDimension = Types.DEFAULT_DNABERT6_PROJECTION_DIMENSION

        if self.projectionState != Types.ProjectionState.NO_PROJECTION and projectionDimension == None:
            self.projectionState = Types.ProjectionState.NO_PROJECTION

        elif self.projectionState == Types.ProjectionState.NOT_TRAINABLE:
            self.projectionDimension = Types.projectionDimension

            # frozen projection (built each call, no grad)
            self.linear = torch.nn.Linear(Types.DEFAULT_DNABERT6_PROJECTION_DIMENSION, self.projectionDimension, bias=False).to(self.device)
            torch.nn.init.xavier_uniform_(self.linear.weight)
            for p in self.linear.parameters():
                p.requires_grad = False

        elif self.projectionState == Types.ProjectionState.TRAINABLE:
            self.projectionDimension = projectionDimension

            #trainable projection
            self.linear = torch.nn.Linear(Types.DEFAULT_DNABERT6_PROJECTION_DIMENSION, self.projectionDimension, bias=False).to(self.device)
            torch.nn.init.xavier_uniform_(self.linear.weight)

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
        self.saveDirectory = saveDir

        self.trainer = None
        self.arguments = None

        print("Initialized DNABERT-6 model for finetuning")
        print(f"Dataset path for finetuning: {self.trainingDataPath}")
        print(f"Dataset percentage to use: {self.trainingDatasetPercentage}%")
        print(f"Learning rate: {self.learningRate}")
        print(f"Window size: {self.windowSize}")
        print(f"Weight decay: {self.weightDecay}")
        print(f"Warmup ratio: {self.warmupRatio}")
        print(f"Finetuning eval barch size: {self.fineTuneEvalBatchSize}")
        print(f"Finetuning train batch size: {self.fineTuneTrainBatchSize}")
        print(f"Embeddings batch size: {self.embeddingsBatchSize}")
        print(f"Projection state: {self.projectionState}")
        print(f"Projection dimension: {self.projectionDimension}")
        print(f"Hidden state: {self.hiddenState}")
        print(f"Directory to save the finetuned model: {self.saveDirectory}")

    def datasetInit(self) -> None:
        #initialize dataset from our previously created csv file

        split = Helpers.loadDatasetPercentage(self.trainingDataPath, self.trainingDatasetPercentage)

        self.trainDataset = split["train"]
        self.validationDataset = split["test"]

        # tokenise every row with DNABERT's tokenizer

        self.trainDataset = self.trainDataset.map(self.encode, batched=True, remove_columns=["sequence"])
        self.validationDataset = self.validationDataset.map(self.encode, batched=True, remove_columns=["sequence"])

        # create PyTorch tensors
        self.trainDataset.set_format(type="torch")
        self.validationDataset.set_format(type="torch")

        print(f"Initialized Training Dataset: {self.trainDataset.shape}")
        print(f"Initialized Validation Dataset: {self.validationDataset.shape}")

    def _poolHidden(self, hidden, attentionMask, state: Types.HiddenState):
        """
        hidden : (B, L, 768) - last_hidden_state from DNABERT
        
        - For state CLS the CLs embeddings from DNABERT6 are returned.
        (B, 768)

        - For state MEAN a mean of all embeddings is calculated excluding 
        embeddings from the CLS token.
        (B, 768)

        - For state BOTH the 2 embeddings of CLS and MEAN are concatenated.
        (B, 1536)
        """

        if state == Types.HiddenState.CLS:

            return hidden[:, 0, :]
        
        elif state == Types.HiddenState.MEAN:

            # mask : (B, L, 1) → 1 for real tokens, 0 for padding
            mask = attentionMask.unsqueeze(-1)

            # exclude [CLS] column (index 0)
            real = hidden[:, 1:, :] * mask[:, 1:, :]     # zero-out pads

            summed = real.sum(1)                           # (B, 768)
            counts = mask[:, 1:, :].sum(1).clamp(min=1e-9) # (B, 1)
            mean = summed / counts                       # (B, 768)

            return mean

        elif state == Types.HiddenState.BOTH:

            # Re-use the two branches above
            cls_vec  = self._poolHidden(hidden, attentionMask, Types.HiddenState.CLS)
            mean_vec = self._poolHidden(hidden, attentionMask, Types.HiddenState.MEAN)

            combined = torch.cat([cls_vec, mean_vec], dim=-1)  # (B, 1536)
            return combined

        else:
            raise ValueError("state must be CLS, MEAN or BOTH")

    def metrics(self, evalPred) -> dict:
        """
        Function given to Trainer to evaluate metrics: 
        accuracy, f1_score, loss, precision.
        """
        # evalPred is usually an EvalPrediction from HF Trainer.
        # In some configs .predictions can be a tuple (e.g., (loss, logits)),
        # so we grab logits in a tuple-safe way.

        logits = (
            evalPred.predictions[0]
            if isinstance(evalPred.predictions, tuple)
            else evalPred.predictions
        )

        labels = evalPred.label_ids # ground-truth class ids (shape [N])

        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels)

        # ── Stable softmax to turn logits into probabilities ──
        # Subtract row-wise max to avoid overflow in exp().

        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Class predictions = argmax over probabilities.

        preds = probs.argmax(axis=1)

        acc  = accuracy_score(labels, preds)
        f1   = f1_score(labels, preds)
        prec, rec, _, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

        # Cross-entropy (log loss) expects PROBABILITIES, not logits.
        # labels=[0,1] fixes the class order even if one class is absent in a batch.

        ce_loss = log_loss(labels, probs,labels=[0, 1])

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "cross_entropy": ce_loss,
        } 

    def encode(self, batch):
        """
        Convert batched string to sequences to kmers and return tokenizer
        """
        batchKmers = [Helpers.kmer(seq, 6, Types.KmerAmbiguousState.MASK) for seq in batch["sequence"]]

        return self.tokenizer(
            batchKmers,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.windowSize
        )

    def finetune(self, **override) -> None:
            """
            Fine-tune the dnabert 6 model, coding and non-coding
            labeled smORFs taken from our train.csv
            """

            self.args = TrainingArguments(
                output_dir                  = self.saveDirectory,
                num_train_epochs            = self.epochs,
                per_device_train_batch_size = self.fineTuneTrainBatchSize,
                per_device_eval_batch_size  = self.fineTuneEvalBatchSize,
                learning_rate               = self.learningRate,
                weight_decay                = self.weightDecay,
                warmup_ratio                = self.warmupRatio,
                eval_strategy               = "epoch",
                save_strategy               = "epoch",
                logging_strategy            = "epoch",
                load_best_model_at_end      = True,
                metric_for_best_model       = "f1",
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
            print(f"✓ Fine-tuned model saved to  {Path(self.saveDirectory).resolve()}")

    def embeddings(self, sequences):
        
        """
        Return an (N, D) NumPy matrix of embeddings. Default size
        of embeddings returned from dnabert6 model is 768.
        """

        self.model.eval()
        self.model.to(self.device)

        vecs = np.empty((len(sequences), self.projectionDimension), dtype=np.float32)
        idx = 0

        with torch.no_grad():

            for i in range(0, len(sequences), self.embeddingsBatchSize):

                batch = sequences[i : i + self.embeddingsBatchSize]
                batchedKmers = [Helpers.kmer(seq, 6, Types.KmerAmbiguousState.MASK) for seq in batch]

                toks  = self.tokenizer(
                    batchedKmers,
                    is_split_into_words=True,
                    truncation=True,
                    padding="max_length",
                    max_length=self.windowSize,
                    return_tensors="pt"
                ).to(self.device)

                attentionMask = toks["attention_mask"]
                                
                hidden = self.model.base_model(**toks).last_hidden_state
                pooled = self._poolHidden(hidden, attentionMask, self.hiddenState)

                if self.linear is not None:
                    pooled = F.relu(self.linear(pooled))

                vecs[idx : idx + pooled.size(0)] = pooled.cpu().numpy()
                idx += pooled.size(0)

        return vecs

    def load(self, modelPath: str) -> None:
        """
        Fully restore a fine-tuned checkpoint:

        • tokenizer.json, vocab.txt, self.tokenizer member variable
        • model.safetensors / pytorch_model, self.model member variable
        • training_args.bin, self.args and other member variables
        """
        path = Path(modelPath)

        # initialize tokenizer and model

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()

        # initialize training arguments

        argsPath = path/"training_args.bin"
        if argsPath.exists():
            self.args = torch.load(argsPath, weights_only=False)

            # pull the key hyper-params back into the object

            self.epochs = int(self.args.num_train_epochs)
            self.learningRate = float(self.args.learning_rate)
            self.weightDecay = float(self.args.weight_decay)
            self.warmupRatio = float(self.args.warmup_ratio)

        else:
            self.args = None

        print("✓ Loaded everything from", path.resolve())

    def history(self) -> pd.DataFrame:

        history = self.trainer.state.log_history
        return pd.DataFrame(history)
    
    def help(self) -> None:

        help(self.model)

    def parameters(self) -> None:

        print(self.model.named_parameters())

    def summary(self) -> None:

        print(self.model)

    def configuration(self) -> None:

        print(self.model.config)