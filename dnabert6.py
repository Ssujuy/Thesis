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

import numpy as np, torch
from pathlib import Path

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

        # tokenise every row with DNABERT's tokenizer

        self.dataset = self.dataset.map(
            self.encode,
            batched=True,
            remove_columns=["sequence"]
        )

        # create PyTorch tensors
        self.dataset.set_format(type="torch")

    def encode(self, batch):
        return self.tokenizer(batch["sequence"], truncation=True, padding="max_length", max_length=self.windowSize)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"f1": f1_score(labels, preds)}

    def finetune(self, outDirectory="dnabert6_smorfs_ft", **override):
            """
            Fine-tune the model.  Override epochs/LR by passing
            finetune(epochs=4, learning_rate=2e-5).
            """

            args = TrainingArguments(
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

            trainer = Trainer(
                self.model,
                args,
                train_dataset               = self.dataset["train"],
                eval_dataset                = self.dataset["validation"],
                data_collator               = DataCollatorWithPadding(self.tokenizer, return_tensors="pt"),
                compute_metrics             = self.compute_metrics,
            )

            trainer.train()
            trainer.save_model(outDirectory)
            self.tokenizer.save_pretrained(outDirectory)
            print(f"✓ Fine-tuned model saved to  {Path(outDirectory).resolve()}")


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