from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    log_loss,
    precision_recall_fscore_support,
)
import pandas as pd
import numpy as np, torch
from pathlib import Path
import json, pickle

### TO TEST
### Load function
### embedding function
### addition for embeddings linear projection
### Mean embeddings withour cls token

class DNABERT6:

    def __init__(
                self,
                modelID: str            = "zhihan1996/DNA_bert_6",
                trainingDataPath: str   = "train.csv",
                epochs: int             = 4,
                learningRate: float     = 2e-5,
                windowSize: int         = 512,
                weightDecay: float      = 0.01,
                warmupRatio: float      = 0.1
            ):

        # tokenizer and model (with classification head)

        self.tokenizer = AutoTokenizer.from_pretrained(modelID)

        # For finding coding - non-coding smORFs we need classification, thus
        # nothing is frozen, backbone + classifier head all have requires_grad=True

        self.model     = AutoModelForSequenceClassification.from_pretrained(modelID, num_labels=2)

        #initialize dataset from our previously created csv file

        self.dataset = load_dataset("csv", data_files={"train": trainingDataPath, "validation": trainingDataPath})

        self.windowSize = windowSize
        self.learningRate = learningRate
        self.epochs = epochs
        self.weightDecay = weightDecay
        self.warmupRatio = warmupRatio

        self.trainer = None
        self.arguments = None

        # tokenise every row with DNABERT's tokenizer

        self.dataset = self.dataset.map(
            self.encode,
            batched=True,
            remove_columns=["sequence"]
        )

        # create PyTorch tensors
        self.dataset.set_format(type="torch")

    def _poolHidden(self, hidden):
        """
        hidden : (B, L, 768)  – last_hidden_state from DNABERT
        """

        return hidden[:, 0, :]

    def metrics(self, eval_pred):
        """
        Function given to Trainer to evaluate metrics: 
        accuracy, f1_score, loss, precision.
        """
        logits, labels = eval_pred
        preds  = np.argmax(logits, axis=-1)

        acc  = accuracy_score(labels, preds)
        f1   = f1_score(labels, preds)
        prec, rec, _, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        ce_loss = log_loss(labels, torch.softmax(torch.tensor(logits), dim=-1))

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "cross_entropy": ce_loss,
        } 

    def encode(self, batch):
        return self.tokenizer(batch["sequence"], truncation=True, padding="max_length", max_length=self.windowSize)

    def embeddings(
            self,
            sequences,
            projectDim = 768,
            batchSize = 32,
            device = "cpu"
            ):
        
        """
        Return an (N, D) NumPy matrix of embeddings.
        """

        self.model.eval()
        self.model.to(device)

        vecs = np.empty((len(sequences), projectDim), dtype=np.float32)

        with torch.no_grad():

            for i in range(0, len(sequences), batchSize):

                batch = sequences[i : i + batchSize]
                toks  = self.tokenizer(
                    batch,
                    truncation=True,
                    padding="max_length",
                    max_length=self.windowSize,
                    return_tensors="pt"
                ).to(device)
                                
                hidden = self.model.base_model(**toks).last_hidden_state
                pooled = self._poolHidden(hidden)

                vecs[idx : idx + pooled.size(0)] = pooled.cpu().numpy()
                idx += pooled.size(0)

        return vecs

    def finetune(self, outDirectory="dnabert6_smorfs_ft", **override):
            """
            Fine-tune the model.  Override epochs/LR by passing
            finetune(epochs=4, learning_rate=2e-5).
            """

            self.args = TrainingArguments(
                output_dir                  = outDirectory,
                num_train_epochs            = self.epochs,
                per_device_train_batch_size = 8,
                per_device_eval_batch_size  = 16,
                learning_rate               = self.learningRate,
                weight_decay                = self.weightDecay,
                warmup_ratio                = self.warmupRatio,
                eval_strategy               = "epoch",
                save_strategy               = "epoch",
                logging_steps               = 10,
                load_best_model_at_end      = True,
                metric_for_best_model       = "f1",
            )

            self.trainer = Trainer(
                self.model,
                self.args,
                train_dataset               = self.dataset["train"],
                eval_dataset                = self.dataset["validation"],
                data_collator               = DataCollatorWithPadding(self.tokenizer, return_tensors="pt"),
                compute_metrics             = self.metrics,
            )

            self.trainer.train()
            self.trainer.save_model(outDirectory)
            self.tokenizer.save_pretrained(outDirectory)
            print(f"✓ Fine-tuned model saved to  {Path(outDirectory).resolve()}")

    def load(self, modelPath: str):
        """
        Fully restore a fine-tuned checkpoint:

        • tokenizer.json, vocab.txt          → self.tokenizer
        • model.safetensors / pytorch_model  → self.model
        • training_args.bin                  → self.args  (and member vars)
        """
        path = Path(modelPath)

        # initialize tokenizer and model

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()

        # initialize training arguments

        ta_path = path / "training_args.bin"
        if ta_path.exists():
            self.args = torch.load(ta_path)

            # pull the key hyper-params back into the object

            self.epochs       = int(self.args.num_train_epochs)
            self.learningRate = float(self.args.learning_rate)
            self.windowSize   = int(self.args.max_length or 512)
            self.weightDecay  = float(self.args.weight_decay)
            self.warmupRatio  = float(self.args.warmup_ratio)

        else:
            self.args = None   # if you saved only the model weights

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
        
model = DNABERT6()
model.finetune()

pd.set_option("display.max_rows", None)    # show all rows
pd.set_option("display.max_columns", None) # show all columns
print(model.history())