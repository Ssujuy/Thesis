import argparse
import csv
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from Bio import SeqIO
from pathlib import Path
import Types
import Helpers

def datasetSplit():
    initialDataset = 'datasets/initial.csv'
    df = pd.read_csv(initialDataset)

    total = len(df)
    coding = len(df[df['label'] == 1])
    nonCoding = len(df[df['label'] == 0])

    pctCoding = coding / total
    pctNonCoding = nonCoding / total

    Helpers.colourPrint(Types.Colours.WHITE,"\n=== Original Dataset Distribution ===")
    Helpers.colourPrint(Types.Colours.WHITE,f"Total: {total}")
    Helpers.colourPrint(Types.Colours.WHITE,f"Coding (1):     {coding} ({pctCoding:.2%})")
    Helpers.colourPrint(Types.Colours.WHITE,f"Non-Coding (0): {nonCoding} ({pctNonCoding:.2%})")

    Helpers.colourPrint(Types.Colours.GREEN,"\nSplitting into 10,000 (finetune.csv) and the rest (training.csv)")

    dfFinetune, dfTrain = train_test_split(
        df, 
        train_size=10000,
        stratify=df['label'],
        random_state=42
    )

    tot = len(dfFinetune)
    cod = len(dfFinetune[dfFinetune['label'] == 1])
    ncod = len(dfFinetune[dfFinetune['label'] == 0])
    Helpers.colourPrint(Types.Colours.WHITE,"\n=== Original Dataset Distribution ===")
    Helpers.colourPrint(Types.Colours.WHITE,f"Total: {tot}")
    Helpers.colourPrint(Types.Colours.WHITE,f"Coding (1):     {cod} ({cod/tot:.2%})")
    Helpers.colourPrint(Types.Colours.WHITE,f"Non-Coding (0): {ncod} ({ncod/tot:.2%})")

    tot = len(dfTrain)
    cod = len(dfTrain[dfTrain['label'] == 1])
    ncod = len(dfTrain[dfTrain['label'] == 0])
    Helpers.colourPrint(Types.Colours.WHITE,"\n=== Original Dataset Distribution ===")
    Helpers.colourPrint(Types.Colours.WHITE,f"Total: {tot}")
    Helpers.colourPrint(Types.Colours.WHITE,f"Coding (1):     {cod} ({cod/tot:.2%})")
    Helpers.colourPrint(Types.Colours.WHITE,f"Non-Coding (0): {ncod} ({ncod/tot:.2%})")

    dfFinetune.to_csv('datasets/finetune.csv', index=False)
    dfTrain.to_csv('datasets/training.csv', index=False)

    Helpers.colourPrint(Types.Colours.GREEN,"\nSuccess! 'finetune.csv' and 'training.csv' have been saved with perfectly preserved class balances.")


def readFastaExtracted(codingFastaExtracted: str, nonCodingFastaExtracted: str, split: int, filepath: str = "datasets/initial.csv"):

    codingTest = "data/extracted/coding_smorfs_2pep_test.fa"
    nonCodingTest = "data/extracted/non_coding_smorfs_2pep_test.fa"

    directory = os.path.dirname(codingTest)

    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    directory = os.path.dirname(nonCodingTest)

    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    totalRecords = sum(1 for _ in SeqIO.parse(codingFastaExtracted, "fasta"))
    recordStop = int(totalRecords * (split / 100))

    Helpers.colourPrint(Types.Colours.PURPLE,f"Filepath: {codingFastaExtracted}, processing: {split}% sequences, {recordStop}/{totalRecords}")

    for index, record in enumerate(SeqIO.parse(codingFastaExtracted, "fasta")):
        if index <= recordStop:
            cleaned = cleanup(record.seq)
            toCsv(cleaned, 1)
        else:
            cleaned = cleanup(record.seq)
            toFasta(cleaned, record.description, codingTest)

    totalRecords = sum(1 for _ in SeqIO.parse(nonCodingFastaExtracted, "fasta"))
    recordStop = int(totalRecords * (split / 100))

    Helpers.colourPrint(Types.Colours.PURPLE,f"Filepath: {nonCodingFastaExtracted}, processing: {split}% sequences, {recordStop}/{totalRecords}")

    for index, record in enumerate(SeqIO.parse(nonCodingFastaExtracted, "fasta")):
        if index <= recordStop:
            cleaned = cleanup(record.seq)
            toCsv(cleaned, 0)
        else:
            cleaned = cleanup(record.seq)
            toFasta(cleaned, record.description, nonCodingTest)

def fastaToList(path: str) -> list:
    """
    Generator that parses DNA sequences from FASTA files, cleans them up and returns them in a list.
    """

    sequenceList = []

    for record in SeqIO.parse(path, "fasta"):
        cleaned = cleanup(record.seq)
        sequenceList.append(cleaned)

    return sequenceList

def readFasta(path: str, label: int) -> None:
    """
    Generator that yields (header, cleaned_seq) for every record in the FASTA.
    """
    for record in SeqIO.parse(path, "fasta"):
        cleaned = cleanup(record.seq)
        toCsv(cleaned, label)

def cleanup(seq: str, size: int = 400) -> str:
    """
    Return an upper-case, ambiguity-free DNA string of exactly pad_to length.
    Truncates if longer.
    """

    seq = str(seq).upper()

    for amb in "NRYWSKMDHBV":
        seq = seq.replace(amb, "N")

    seqLen = len(seq)
    
    if seqLen > size:
        seq = seq[:size]

    elif seqLen < size:
        noise = ''.join(random.choices("ATGC", k=(size - seqLen)))
        seq += noise

    return seq

def toFasta(sequence:str, header: str, fastapath: str):

    with open(fastapath, 'a') as file:
        file.write(f">{header}\n{sequence}\n")

def toCsv(sequence: str, label: int, filepath: str = "datasets/initial.csv") -> None:

    filepath = Path(filepath)
    exists = filepath.exists()

    with filepath.open("a", newline="") as file:

        writer = csv.writer(file)

        if not exists:
            writer.writerow(["sequence", "label"])
        
        writer.writerow([sequence, label])

def main(codingFasta100Flank: str, nonCodingFasta100Flank: str, codingFastaExtracted: str, nonCodingFastaExtracted):
    
    Helpers.colourPrint(Types.Colours.PURPLE,f"Reading FASTA file for coding smORFs: {codingFasta100Flank} and non-coding smORFs: {nonCodingFasta100Flank}")
    readFasta(codingFasta100Flank, 1)
    readFasta(nonCodingFasta100Flank, 0)
    Helpers.colourPrint(Types.Colours.PURPLE,f"Reading FASTA file for coding smORFs: {codingFastaExtracted} and non-coding smORFs: {nonCodingFastaExtracted}")
    readFastaExtracted(codingFastaExtracted,nonCodingFastaExtracted,70)

    datasetSplit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--codingFasta100Flank", type=str, default="data/100flank/positive_trainingSet_Flank-100.fa",
                        help="Path to the input FASTA file for coding smORFs with 100flank")
    parser.add_argument("--nonCodingFasta100Flank", type=str, default="data/100flank/negative_trainingSet_Flank-100.fa",
                        help="Path to the input FASTA file for non-coding smORFs with 100flank")
    parser.add_argument("--codingFastaExtracted", type=str, default="data/extracted/coding_smorfs_2pep_5890.fa",
                        help="Path to the input FASTA file for coding smORFs extracted from OpenProt, NCBI and Ensembl")
    parser.add_argument("--nonCodingFastaExtracted", type=str, default="data/extracted/non_coding_smorfs_2pep_5890.fa",
                        help="Path to the input FASTA file for non-coding smORFs extracted from OpenProt, NCBI and Ensembl")
    args = parser.parse_args()
    
    main(
        codingFasta100Flank=args.codingFasta100Flank,
        nonCodingFasta100Flank=args.nonCodingFasta100Flank,
        codingFastaExtracted=args.codingFastaExtracted,
        nonCodingFastaExtracted=args.nonCodingFastaExtracted
    )
