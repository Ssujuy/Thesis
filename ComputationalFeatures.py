import torch
import numpy as np

from HexamerScore import HexamerScore
from FickettScore import FicketScore
from CodonBias import CodonBias
from NucleotideBias import NucleotideBias

class ComputationalFeatures:

    def __init__(self, trainPath: str):

        self.hexamerScore = HexamerScore(trainPath)
        self.ficketScore = FicketScore(trainPath)
        self.codonBias = CodonBias(trainPath)
        self.nucleotideBias = NucleotideBias(trainPath)

        self.input = 4

    def score(self, sequences) -> list:
        batch_scores = []

        # Iterate through the sequences
        for seq in sequences:
            s1 = self.hexamerScore.score(seq)
            s2 = self.ficketScore.score(seq)
            s3 = self.codonBias.score(seq)
            s4 = self.nucleotideBias.score(seq)
            
            batch_scores.append([s1, s2, s3, s4])

        # Convert efficiently to Tensor via numpy to ensure correct dtype
        return torch.tensor(np.array(batch_scores), dtype=torch.float32)