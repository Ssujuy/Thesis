from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset
import evaluate, numpy as np, torch

class DNABERT6:

    def __init__(self, modelID: str = "zhihan1996/DNA_bert_6", trainingDataPath: str = "train.csv"):

        # tokenizer and model (with classification head)
        self.tokenizer = AutoTokenizer.from_pretrained(modelID)
        self.model     = AutoModelForSequenceClassification.from_pretrained(modelID, num_labels=2)
        self.dataset = load_dataset("csv", data_files={"train": trainingDataPath},column_names=["sequence", "label"])

        # tokenise every row with DNABERT's tokenizer
        self.dataset = self.dataset.map(
            self.encode,
            batched=True,
            remove_columns=["sequence"]
        )

        # create PyTorch tensors
        self.dataset.set_format(type="torch")

    def encode(self, batch):
        return self.tokenizer(batch["sequence"], truncation=True, padding="max_length", max_length=512)
    
    def finetune(self):
        return None
    

    def help(self) -> None:

        help(self.model)

    def parameters(self) -> None:

        print(self.model.named_parameters())

    def summary(self) -> None:

        print(self.model)

    def configuration(self) -> None:

        print(self.model.config)
        
model = DNABERT6()

model.parameters()