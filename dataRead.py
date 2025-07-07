import argparse
import csv
from Bio import SeqIO
from pathlib import Path
from computationalFeatures import firstORFLength

DEFAULT_WINDOW_SIZE = 512          # fixed length 512 for DNABERT / CNN

### are padding only on the right??

def readFasta(path: str, label: int) -> None:
    """
    Generator that yields (header, cleaned_seq) for every record in the FASTA.
    """
    for record in SeqIO.parse(path, "fasta"):
        cleaned = cleanup(record.seq)
        toCsv(cleaned, label)

    

def cleanup(seq: str, size: int = DEFAULT_WINDOW_SIZE) -> str:
    """
    Return an upper-case, ambiguity-free DNA string of exactly pad_to length.
    Truncates if longer, pads with 'N' tokens if shorter.
    """

    seq = str(seq).upper()

    # replace every ambiguity symbol with 'N'
    for amb in "NRYWSKMDHBV":
        seq = seq.replace(amb, "N")

    # pad or truncate to fixed window size
    if len(seq) > size:
        seq = seq[:size]
    else:
        seq = seq + "N" * (size - len(seq))

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
    parser.add_argument("codingFASTA", type=str,
                        help="Path to the input FASTA file for coding smORFs")
    parser.add_argument("nonCodingFASTA", type=str,
                        help="Path to the input FASTA file for non-coding smORFs")
    
    args = parser.parse_args()
    
    main(codingFasta=args.codingFASTA, nonCodingFasta=args.nonCodingFASTA)
