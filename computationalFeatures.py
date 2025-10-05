import itertools
import pandas as pd
import math

import Helpers

class HexamerScore():
    def __init__(self, sequencesPath: str):
        self.k = 6
        self.bases = ['A', 'C', 'G', 'T']

        hexamers = [''.join(kmer) for kmer in itertools.product(self.bases, repeat=self.k)]

        self.codingHexamersFreq = {h:0 for h in hexamers}
        self.nonCodingHemaerFreq = {h:0 for h in hexamers}

        df = pd.read_csv(sequencesPath)

        for sequence, label in zip(df["sequence"], df["label"]):
            if 'N' not in sequence:
                if label == 1:
                    kmerSequence = Helpers.kmer(sequence, self.k)
                    for kmer in kmerSequence:
                        self.codingHexamersFreq[kmer] += 1
                else:
                    kmerSequence = Helpers.kmer(sequence, self.k)
                    for kmer in kmerSequence:
                        self.nonCodingHemaerFreq[kmer] += 1

        totalCoding = sum(self.codingHexamersFreq.values()) + 4**self.k * 1.0
        totalNonCoding = sum(self.nonCodingHemaerFreq.values()) + 4**self.k * 1.0

        for key, value in self.codingHexamersFreq.items():
            self.codingHexamersFreq[key] = (value + 1.0) / totalCoding

        for key, value in self.nonCodingHemaerFreq.items():
            self.nonCodingHemaerFreq[key] = (value + 1.0) / totalNonCoding

    def score(self, sequence: str) -> float:
        total = 0.0
        p = 0.0
        kmerSequence = Helpers.kmer(sequence, self.k)

        for kmer in  kmerSequence:
            p += math.log(self.codingHexamersFreq[kmer] / self.nonCodingHemaerFreq[kmer])
            total += 1

        return p / total
    
class FicketScore():

    def __init__(self, sequencePath: str):

def ficketScore(sequence: str):

    frame1 = sequence[0::3]
    frame2 = sequence[1::3]
    frame3 = sequence[2::3]

    length = len(sequence)

    frequencies = {b: sequence.count(b) / length for b in "ATCG"}

    fpos = {b: [frame1.count(b)/len(frame1),
            frame2.count(b)/len(frame2),
            frame3.count(b)/len(frame3)] for b in "ATCG"}
    
    bias = {b: max(fpos[b]) / (min(fpos[b]) + 1e-9) for b in "ATCG"}

    fickett = sum((frequencies[b] + bias[b]) / 2 for b in "ATCG") / 4

    return fickett

seq1 = "ATGCGTACGTTGCGACCTAGTGACGTGACCATGCGTATCGTGACGATCGTACGTAGCTAGCTG"
seq2 = "CGTAGCTAGGCTAACGTTGACGATGCGTACGCTTAGCATGACCGATCGTAGCATCGTACGAT"
seq3 = "GCTAGCATGACGATCGTACGTAGCATGCGTACGATCGTAGCTAGGCTAGCATGACCGTACGA"

print("Score 1:", ficketScore(seq1))
print("Score 2:", ficketScore(seq2))
print("Score 3:", ficketScore(seq3))
