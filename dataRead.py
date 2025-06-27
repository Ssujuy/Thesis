import argparse
import csv
from Bio import SeqIO
from pathlib import Path

DEFAULT_WINDOW_SIZE = 512          # fixed length 512 for DNABERT / CNN

def readFasta(path: str):
    """
    Generator that yields (header, cleaned_seq) for every record in the FASTA.
    """
    for record in SeqIO.parse(path, "fasta"):
        cleaned = cleanup(record.seq)
        return record.id, cleaned

    

def cleanup(seq: str, size: int = DEFAULT_WINDOW_SIZE):
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

def toCsv(label: int, sequence: str, filepath: str = "train.csv"):

    filepath = Path(filepath)
    exists = filepath.exists()

    with filepath.open("a", newline="") as file:

        writer = csv.writer(file)

        if not exists:
            writer.writerow(["label", "sequence"])
        
        writer.writerows([label, sequence])

def main(filepath: str = ""):
    
    print(f"Reading FASTA file: {filepath}")
    id,seq = readFasta(filepath)
    print(seq + "..." )
    print(f"length = {len(seq)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str,
                        help="Path to the input FASTA file")
    
    args = parser.parse_args()
    
    main(filepath=args.filepath)
