import itertools
import pandas as pd
import numpy as np
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
        self.nonCodingHemaerFreq = {h:0 for h in hexamers}

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
                        self.nonCodingHemaerFreq[kmer] += 1

        totalCoding = sum(self.codingHexamersFreq.values()) + 4**self.k * 1.0
        totalNonCoding = sum(self.nonCodingHemaerFreq.values()) + 4**self.k * 1.0

        for key, value in self.codingHexamersFreq.items():
            self.codingHexamersFreq[key] = (value + 1.0) / totalCoding

        for key, value in self.nonCodingHemaerFreq.items():
            self.nonCodingHemaerFreq[key] = (value + 1.0) / totalNonCoding

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
            p += math.log(self.codingHexamersFreq[kmer] / self.nonCodingHemaerFreq[kmer])
            total += 1

        return p / total
    
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

                if label == 1:
                    self.codingProb[str(p)][base] += 1
                else:
                    self.nonCodingProb[str(p)][base] += 1

        for p in self.positions:

            totalCodingInPos = sum(self.codingProb[str(p)].values())
            totalNonCodingInPos = sum(self.nonCodingProb[str(p)].values())

            if totalCodingInPos == 0: totalCodingInPos = 1
            if totalNonCodingInPos == 0: totalNonCodingInPos = 1

            for b in self.bases:
                self.codingProb[str(p)][b] = (self.codingProb[str(p)][b] + 1) / (totalCodingInPos + 4) 
                self.nonCodingProb[str(p)][b] = (self.nonCodingProb[str(p)][b] + 1) / (totalNonCodingInPos + 4)

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

                frac = self.codingProb[str(p)][base] / (self.nonCodingProb[str(p)][base] + 1e-9)

                if frac > 0.0:
                    score += math.log(frac)
                else:
                    score -= 10.0

        return score
    
class CodonBias:
    """
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

        self.tis = "ATG"
        self.codonLen = 3

        self.codonToAminoAcid = {
            'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
            'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
            'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
            'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
            'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
            'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
            'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
            'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
            'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
            'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
            'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
            'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
            'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
            'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
            'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
            'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
        }

        df = pd.read_csv(sequencesPath)

        aminoAcids = set(self.codonToAminoAcid.values())
        codons = self.codonToAminoAcid.keys()

        codingCodonCounts = {c:0 for c in codons}
        nonCodingCodonCounts = {c:0 for c in codons}

        codingAminoAcidCounts = {aa: 0 for aa in aminoAcids}
        nonCodingAminoAcidCounts = {aa: 0 for aa in aminoAcids}

        for sequence, label in zip(df["sequence"], df["label"]):

            if len(sequence) < self.codonLen:
                continue

            index = sequence.find(self.tis)

            if index == -1:
                continue
            
            orfSequence = sequence[index:]
            orfSeqLength = (len(orfSequence) // 3) * 3
            orfSequence = orfSequence[:orfSeqLength]

            for i in range(0, len(orfSequence) , 3):

                codon = orfSequence[i:i+3]

                if codon in self.codonToAminoAcid:

                    aminoAcid = self.codonToAminoAcid[codon]

                    if aminoAcid == '_':
                        continue

                    if label == 1:
                        codingCodonCounts[codon] += 1
                        codingAminoAcidCounts[aminoAcid] += 1

                    else:
                        nonCodingCodonCounts[codon] += 1
                        nonCodingAminoAcidCounts[aminoAcid] += 1

        self.codingProb = {}
        self.nonCodingProb = {}

        for codon in codons:

            aminoAcid = self.codonToAminoAcid[codon]
            
            if aminoAcid == '_':
                continue

            codingCodonCount = codingCodonCounts[codon]
            codingAminoAcidCount = codingAminoAcidCounts[aminoAcid]
            self.codingProb[codon] = codingCodonCount / (codingAminoAcidCount + 1e-9)

            nonCodingCodonCount = nonCodingCodonCounts[codon]
            nonCodingAminoAcidCount = nonCodingAminoAcidCounts[aminoAcid]
            self.nonCodingProb[codon] = nonCodingCodonCount / (nonCodingAminoAcidCount + 1e-9)

    def score(self, sequence: str):
        """
        """

        codonBias = 0.0
        codonsCount = 0

        if len(sequence) < self.codonLen:
            raise ValueError(f"Given sequence has less than {self.codonLen} bases")

        index = sequence.find(self.tis)
        if index == -1:
            return 0.0

        orfSequence = sequence[index:]
        orfSeqLength = (len(orfSequence) // 3) * 3
        orfSequence = orfSequence[:orfSeqLength]

        for i in range(0, len(orfSequence), 3):
            codon = orfSequence[i:i+3]
            
            if codon in self.codonToAminoAcid:
                
                aminoAcid = self.codonToAminoAcid[codon]
                if aminoAcid == '_':
                    continue

                codingPCodon = self.codingProb[codon]
                nonCodingPCodon = self.nonCodingProb[codon]

                if codingPCodon == 0:
                    codingPCodon = 1e-9

                codonBias += math.log(codingPCodon / nonCodingPCodon)
                codonsCount += 1

        return codonBias / codonsCount