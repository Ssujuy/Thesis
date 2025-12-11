import itertools
import pandas as pd
import numpy as np
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

    def __init__(self, sequencesPath: str):

        self.bases = ['A', 'C', 'G', 'T']
        self.codingFickettStats = {}
        self.nonCodingFickettStats = {}

        codingFeatures = []
        nonCodingFeatures = []

        df = pd.read_csv(sequencesPath)

        for sequence, label in zip(df["sequence"], df["label"]):

            baseFreq = self._baseFrequencies(sequence)
            framesFreq = self._frameFrequencies(sequence)
            bias = self._calculatePeriodicityBias(framesFreq)

            features = []

            for b in self.bases:
                features.append(baseFreq[b])

            for b in self.bases:
                features.append(bias[b])

            if label == 1:
                codingFeatures.append(features)
            else:
                nonCodingFeatures.append(features)

        coding_features_np = np.array(codingFeatures)
        noncoding_features_np = np.array(nonCodingFeatures)

        self.codingFickettStats['mean'] = coding_features_np.mean(axis=0)
        self.codingFickettStats['std'] = coding_features_np.std(axis=0) + 1e-9 
        
        self.nonCodingFickettStats['mean'] = noncoding_features_np.mean(axis=0)
        self.nonCodingFickettStats['std'] = noncoding_features_np.std(axis=0) + 1e-9

    def _baseFrequencies(self, sequence: str) -> dict:

        frequencies = {}
        sequenceLength = len(sequence)

        for b in self.bases:

            frequence = sequence.count(b) / sequenceLength
            frequencies[b] = frequence

        return frequencies

    def _frameFrequencies(self, sequence: str) -> dict:

        frame1 = sequence[0::3]
        frame2 = sequence[1::3]
        frame3 = sequence[2::3]

        frequencies = {}

        for b in self.bases:

            frame1Freq = frame1.count(b) / len(frame1) if len(frame1) > 0 else 0
            frame2Freq = frame2.count(b) / len(frame2) if len(frame2) > 0 else 0
            frame3Freq = frame3.count(b) / len(frame3) if len(frame3) > 0 else 0

            frequencies[b] = [frame1Freq, frame2Freq, frame3Freq]

        return frequencies
    
    def _calculatePeriodicityBias(self, frequencies: dict) -> dict:

        bias = {}

        for b in self.bases:

            maxFreq = max(frequencies[b])
            minFreq = [f for f in frequencies[b] if f > 0]

            if not minFreq:
                minFreq = 1e-9

            else:
                minFreq = min(minFreq)

            bias[b] = maxFreq / minFreq

        return bias

    def score(self, sequence: str) -> float:
        """
        Calculates the Fickett Score for a given sequence by comparing its\n
        raw features against the trained coding and non-coding distributions.\n
        This method returns a single scalar score, that acts as a strong feature for the classifier.
        """

        if len(sequence) < 3:
            raise ValueError("Sequence given is less than 3 nucleotide length")

        baseFreq = self._baseFrequencies(sequence)
        framesFreq = self._frameFrequencies(sequence)
        bias = self._calculatePeriodicityBias(framesFreq)
        
        features = []
        for b in self.bases:
            features.append(baseFreq[b])
        for b in self.bases:
            features.append(bias[b])

        score = 0.0

        for i in range(len(features)):

            value = features[i]
            zNonCoding = (value - self.nonCodingFickettStats['mean'][i]) / self.nonCodingFickettStats['std'][i]
            zCoding = (value - self.codingFickettStats['mean'][i]) / self.codingFickettStats['std'][i]
            score += zNonCoding - zCoding

        return score / len(features)