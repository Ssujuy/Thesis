import pandas as pd
import numpy as np

class FicketScore():
    """
    Class that initializes look-up tables from coding and non-coding small open reading frame DNA samples for Fickett Score calculation.
    Look-up tables are initialized by calculating each base frequencies and periodicity bias by splitting the sequence into 3 frames.
    Then the frequency of each base is calculated for each frame and the stats are appended to their designated coding potential (coding vs non-coding).
    
    Attributes
    ----------  
    bases : list
        List of valid bases inside a DNA sequence.
    
    codingFickettStats : dict
        Dictionary containing `mean` and `std` calculations of coding Fickett stats (base frequencies and periodicity biases).

    nonCodingHexamersFreq : dict
        Dictionary containing `mean` and `std` calculations of non-coding Fickett stats (base frequencies and periodicity biases).

    Methods
    ----------
    _baseFrequencies(self, sequence: str) -> dict:
        Calculates frequency for each base on a given sequence.
    
    _frameFrequencies(self, sequence: str) -> dict:    
        Calculates frequency for each base on a frame of a sequence.
    
    _calculatePeriodicityBias(self, frequencies: dict) -> dict:
        Calculates periodicity bias (maximum frequency / minimum frequency : for each base).
    
    score(sequence: str) -> float
        Calculates and returns Fickett Score of a given DNA sequence.
    """

    def __init__(self, sequencesPath: str):
        """
        Constructs look-up tables that calculate fickett stats, from known coding and non-coding smORF samples.

        Parameters
        ----------
        sequencesPath : str
            Path to csv file containing [sequence, label] columns of known coding and non-coding sequences.
        """

        self.bases = ['A', 'C', 'G', 'T']
        self.codingFickettStats = {}
        self.nonCodingFickettStats = {}

        codingFeatures = []
        nonCodingFeatures = []

        df = pd.read_csv(sequencesPath)

        for sequence, label in zip(df["sequence"], df["label"]):

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

            if label == 1:
                codingFeatures.append(features)
            else:
                nonCodingFeatures.append(features)

        codingFeatures = np.array(codingFeatures)
        nonCodingFeatures = np.array(nonCodingFeatures)

        self.codingFickettStats['mean'] = codingFeatures.mean(axis=0)
        self.codingFickettStats['std'] = codingFeatures.std(axis=0) + 1e-9 
        
        self.nonCodingFickettStats['mean'] = nonCodingFeatures.mean(axis=0)
        self.nonCodingFickettStats['std'] = nonCodingFeatures.std(axis=0) + 1e-9

    def _baseFrequencies(self, sequence: str) -> dict:
        """
        Calculates frequency for each base on a given sequence.
        Returns dictionary with bases as keys and frequencies as values.
        
        Parameters
        ----------
        sequence : str
            DNA sequence as string.

        Return
        ----------
        dict
            Dictionary containing frequencies for all bases.
        """

        frequencies = {}
        sequenceLength = len(sequence)

        for b in self.bases:

            frequence = sequence.count(b) / sequenceLength
            frequencies[b] = frequence

        return frequencies

    def _frameFrequencies(self, sequence: str) -> dict:
        """
        Calculates frequencies for all bases, on a seqeunce split into 3 frames.

        Parameters
        ----------
        sequence : str
            DNA sequence as string.

        Return
        ----------
        dict
            Dictionary containing frequencies for all bases for all 3 frames.
        """

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
        """
        Calculates periodicity bias for all bases from the frequencies produced by the 3 frames.

        Parameters
        ----------
        frequencies : dict
            Contains frequences for each base and for each frame.

        Return
        ----------
        dict
            Dictionary containing periodicity bias for each base.
        """

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
        This method returns a single scalar score.

        Parameters
        ----------
        sequence : str
            DNA sequence as string.

        Return
        ----------
        float
            Fickett Score for given DNA sequence.
        """

        if len(sequence) < 3:
            raise ValueError("Sequence given is less than 3 length")

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