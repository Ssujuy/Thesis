from transformers import AutoTokenizer, AutoModel

class DNABERT6:

    def __init__(self, modelID="zhihan1996/DNA_bert_6"):

        self.tokenizer = AutoTokenizer.from_pretrained(modelID)
        self.model = AutoModel.from_pretrained(modelID)

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