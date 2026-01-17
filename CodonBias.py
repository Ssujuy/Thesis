import pandas as pd
import math

class CodonBias:
    """
    Class that initializes dictionaries from coding and non-coding small open reading frame DNA samples for Codon Bias calculation.
    Look-up dictionaries are used to calculate coding and non coding probabillity of codons, which is calculated by the ratio of codon frequency and amino acid frequency.
    
    Attributes
    ----------
    tis : str
        Translation Initiation Site.
    
    codonLen : int
        Length of codons.
    
    codonToAminoAcid : dict
        Dictionary with all possible codons as keys and their produced amino acids as values.

    codingProb : dict
        Dictionary with codons as keys and their saved probability (codonCount / amino acid count), for coding smORFs.

    nonCodingProb : dict
        Dictionary with codons as keys and their saved probability (codonCount / amino acid count), for non-coding smORFs.        

    Methods
    ----------
    score(sequence: str) -> float
        Calculates and returns Codon Bias of a given DNA sequence. 
    """

    def __init__(self, sequencesPath: str):
        """
        Initializes codon to amino acid dictionary, imports the csv DataFrame and counts codons and amino acids.
        Dictionaries, for coding and non-coding smORFS, are create with codons are keys and values the probability,
        calculated by the count of codons divided by the count of amino acids

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

            if self.codingProb[codon] == 0.0:
                self.codingProb[codon] = 1e-9

            nonCodingCodonCount = nonCodingCodonCounts[codon]
            nonCodingAminoAcidCount = nonCodingAminoAcidCounts[aminoAcid]
            self.nonCodingProb[codon] = nonCodingCodonCount / (nonCodingAminoAcidCount + 1e-9)

            if self.nonCodingProb[codon] == 0.0:
                self.nonCodingProb[codon] = 1e-9

    def score(self, sequence: str):
        """
        Calculates the Codon Bias of a DNA sequence using pre-initialized dictionaries from labeled smORFs.
        
        Parameters
        ----------
        sequence : str
            DNA sequence as string.

        Return
        ----------
        float
            Codon Bias score of the given sequence.
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

                codonBias += math.log(self.codingProb.get(codon, 1e-9) / self.nonCodingProb.get(codon, 1e-9))
                codonsCount += 1

        return codonBias / codonsCount