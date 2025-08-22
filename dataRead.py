import argparse
import csv
from Bio import SeqIO
from pathlib import Path
from computationalFeatures import firstORFLength
import Types

def readFasta(path: str, label: int) -> None:
    """
    Generator that yields (header, cleaned_seq) for every record in the FASTA.
    """
    for record in SeqIO.parse(path, "fasta"):
        cleaned = cleanup(record.seq)
        toCsv(cleaned, label)

def cleanup(seq: str, size: int = Types.DEFAULT_DNABERT6_WINDOW_SIZE) -> str:
    """
    Return an upper-case, ambiguity-free DNA string of exactly pad_to length.
    Truncates if longer.
    """

    seq = str(seq).upper()

    # replace every ambiguity symbol with 'N'
    for amb in "NRYWSKMDHBV":
        seq = seq.replace(amb, "N")

    # truncate to fixed window size
    if len(seq) > size:
        seq = seq[:size]

    return seq

def toCsv(sequence: str, label: int, filepath: str = "train.csv") -> None:

    filepath = Path(filepath)
    exists = filepath.exists()

    with filepath.open("a", newline="") as file:

        writer = csv.writer(file)

        if not exists:
            writer.writerow(["sequence", "label"])
        
        writer.writerow([sequence, label])

def main(codingFasta: str = "", nonCodingFasta: str = ""):
    
    print(f"Reading FASTA file for coding smORFs: {codingFasta} and non-coding smORFs: {nonCodingFasta}")
    readFasta(codingFasta, 1)
    readFasta(nonCodingFasta, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--codingFastaPath", type=str, required=True,
                        help="Path to the input FASTA file for coding smORFs")
    parser.add_argument("--nonCodingFastaPath", type=str, required=True,
                        help="Path to the input FASTA file for non-coding smORFs")
    
    args = parser.parse_args()
    
    main(codingFasta=args.codingFastaPath, nonCodingFasta=args.nonCodingFastaPath)
