import itertools
import pandas as pd
import math

import Helpers

class HexamerScore():
    """
    Class that initializes look-up tables from coding and non-coding small open reading frame DNA samples for Hexamer Score calculation.
    Look-up tables are used to calculate Hexamer score score of any given DNA sequence using `score` function.
    
    Attributes
    ----------
    k : int
        Number k of k-mer (split of DNA sequence).
    
    bases : list
        List of valid bases inside a DNA sequence.
    
    codingHexamersFreq : dict
        Dictionary for coding smORFs with bases as keys and frequency as value.

    nonCodingHexamersFreq : dict
        Dictionary for non-coding smORFs with bases as keys and frequency as value.

    Methods
    ----------
    score(sequence: str) -> float
        Calculates and returns Hexamer Score of a given DNA sequence. 
    """

    def __init__(self, sequencesPath: str):
        """
        Constructs look-up tables that calculate hexamer frequencies, from known coding and non-coding smORF samples.

        Parameters
        ----------
        sequencesPath : str
            Path to csv file containing [sequence, label] columns of known coding and non-coding sequences.
        """

        self.k = 6
        self.bases = ['A', 'C', 'G', 'T']

        hexamers = [''.join(kmer) for kmer in itertools.product(self.bases, repeat=self.k)]

        self.codingHexamersFreq = {h:0 for h in hexamers}
        self.nonCodingHexamersFreq = {h:0 for h in hexamers}

        df = pd.read_csv(sequencesPath)

        for sequence, label in zip(df["sequence"], df["label"]):

            if len(sequence) < self.k:
                raise ValueError(f"Sequence given is less than {self.k} length")

            if 'N' not in sequence:
                if label == 1:
                    kmerSequence = Helpers.kmer(sequence, self.k)
                    for kmer in kmerSequence:
                        self.codingHexamersFreq[kmer] += 1
                else:
                    kmerSequence = Helpers.kmer(sequence, self.k)
                    for kmer in kmerSequence:
                        self.nonCodingHexamersFreq[kmer] += 1

        totalCoding = sum(self.codingHexamersFreq.values())
        totalNonCoding = sum(self.nonCodingHexamersFreq.values())

        for key, value in self.codingHexamersFreq.items():

            self.codingHexamersFreq[key] = value / totalCoding

            if self.codingHexamersFreq[key] == 0:
                self.codingHexamersFreq[key] = 1e-9

        for key, value in self.nonCodingHexamersFreq.items():

            self.nonCodingHexamersFreq[key] = value / totalNonCoding

            if self.nonCodingHexamersFreq[key] == 0:
                self.nonCodingHexamersFreq[key] = 1e-9

    def score(self, sequence: str) -> float:
        """
        Calculates the Hexamer Score of a given DNA sequence, using initialized look-up tables.

        Parameters
        ----------
        sequence : str
            DNA sequence given to calculate its Hexamer Score.
        
        Return
        ----------
        float
            Calculated Hexamer Score.
        """

        if len(sequence) < self.k:
            raise ValueError(f"Sequence given is less than {self.k} length")

        total = 0.0
        p = 0.0
        kmerSequence = Helpers.kmer(sequence, self.k)

        for kmer in  kmerSequence:

            p += math.log(self.codingHexamersFreq.get(kmer, 1e-9) / self.nonCodingHexamersFreq.get(kmer, 1e-9))
            total += 1

        return p / total