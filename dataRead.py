import argparse
import csv
from Bio import SeqIO
from pathlib import Path

DEFAULT_WINDOW_SIZE = 512          # fixed length 512 for DNABERT / CNN

### we need to add a logic in this to correctly separate
### positive from negative labels, probably from file path
### non-coding smORFs ---> negative/sequences.fa and
### coding smORFs ---> positive/sequences.fa

def readFasta(path: str) -> None:
    """
    Generator that yields (header, cleaned_seq) for every record in the FASTA.
    """
    for record in SeqIO.parse(path, "fasta"):
        cleaned = cleanup(record.seq)
        toCsv(cleaned, 1)

    

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

def toCsv(sequence: str, label: str, filepath: str = "train.csv") -> None:

    filepath = Path(filepath)
    exists = filepath.exists()

    with filepath.open("a", newline="") as file:

        writer = csv.writer(file)

        if not exists:
            writer.writerow(["Sequence", "Label"])
        
        writer.writerow([sequence, label])

def main(filepath: str = ""):
    
    print(f"Reading FASTA file: {filepath}")
    readFasta(filepath)
    # print(seq + "..." )
    # print(f"length = {len(seq)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str,
                        help="Path to the input FASTA file")
    
    args = parser.parse_args()
    
    main(filepath=args.filepath)
