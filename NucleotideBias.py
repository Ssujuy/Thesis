import pandas as pd
import math

class NucleotideBias():
    """
    Implements a class used to calculate Nucleotide Bias of any given sequence.
    At initialization, dictionaries are created from a sample of non-coding and coding smORFs.
    The dictionaries contain probabillities of each base `['A', 'C', 'G', 'T']` existing in each position `[-3, -2, -1, 3, 4, 5]`.
    Where 0,1,2 positions are the 'ATG' codon translation initiation site (TIS).
    If no ATG is not found score is undefined (0.0), if the positions do not exist the position is ignored.

    Attributes
    ----------  
    bases : list
        List of valid bases inside a DNA sequence.

    positions : list
        List of positions before and after 'ATG'.

    tis : str
        Translation Initiation Site codon.
        
    codingProb : dict
        Dictionary containing probabillities of each base existing in each position for coding smORFs.

    nonCodingProb : dict
        Dictionary containing probabillities of each base existing in each position for non-coding smORFs.

    Methods
    ----------  
    score(sequence: str) -> float
        Calculates and returns Nucleotide Bias score of a given DNA sequence.
    """

    def __init__(self, sequencesPath: str):
        """
        Constructs dictionaries codingProb and nonCodingProb from lebaled smORFs dataset located at: `sequencesPath`.
        codingProb dictionary contains probabillity of a base existing in each position, in coding smORFs.
        nonCodingProb dictionary contains probabillity of a base existing in each position, in non-coding smORFs.

        Parameters
        ----------
        sequencesPath : str
            Path to csv file containing [sequence, label] columns of known coding and non-coding sequences.
        """

        self.bases = ['A', 'C', 'G', 'T']
        self.positions = [-3, -2, -1, 3, 4, 5]
        self.tis = "ATG"

        df = pd.read_csv(sequencesPath)

        self.codingProb = {}
        self.nonCodingProb = {}

        for p in self.positions:
            self.codingProb[str(p)] = {}
            self.nonCodingProb[str(p)] = {}            

        for p in self.positions:
            for b in self.bases:
                self.codingProb[str(p)][b] = 0
                self.nonCodingProb[str(p)][b] = 0
    
        for sequence, label in zip(df["sequence"], df["label"]):

            if len(sequence) < len(self.positions) + len(self.tis):
                print(f"Sequence length size < {len(self.tis) + len(self.positions)}, skipping...")
                continue

            index = sequence.find(self.tis)

            if index == -1:
                print(f"No ATG was found, skipping...")
                continue

            for p in self.positions:

                positionIndex = index + p

                if positionIndex < 0 or positionIndex >= len(sequence):
                    continue

                base = sequence[positionIndex]

                if base not in self.bases:
                    continue

                if label == 1:
                    self.codingProb[str(p)][base] += 1
                else:
                    self.nonCodingProb[str(p)][base] += 1

        for p in self.positions:

            totalCodingInPos = sum(self.codingProb[str(p)].values())
            totalNonCodingInPos = sum(self.nonCodingProb[str(p)].values())

            if totalCodingInPos == 0:
                totalCodingInPos = 1.0

            if totalNonCodingInPos == 0:
                totalNonCodingInPos = 1.0

            for b in self.bases:
                self.codingProb[str(p)][b] = self.codingProb[str(p)][b] / totalCodingInPos
                self.nonCodingProb[str(p)][b] = self.nonCodingProb[str(p)][b] / totalNonCodingInPos

                if self.codingProb[str(p)][b] == 0.0:
                    self.codingProb[str(p)][b] = 1e-9

                if self.nonCodingProb[str(p)][b] == 0.0:
                    self.nonCodingProb[str(p)][b] = 1e-9

    def score(self, sequence: str) -> float:
        """
        Calculates the Nucleotide Bias of a DNA sequence using pre-initialized dictionaries from labeled smORFs.
        
        Parameters
        ----------
        sequence : str
            DNA sequence as string.

        Return
        ----------
        float
            Nucleotide Bias score of the given sequence.
        """

        if len(sequence) < len(self.tis):
            raise ValueError(f"Given sequence has less than {len(self.tis)} bases")

        score = 0.0
        index = sequence.find(self.tis)

        if index == -1:
            return score
        
        for p in self.positions:

            positionIndex = index + p

            if positionIndex < 0 or positionIndex >= len(sequence):
                continue

            base = sequence[positionIndex]

            if base in self.bases:
                score += math.log(self.codingProb[str(p)][base] / self.nonCodingProb[str(p)][base])

        return score