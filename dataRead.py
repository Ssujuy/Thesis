import argparse
from Bio import SeqIO

def read(filepath):
    for seq in SeqIO.parse(handle=filepath, format="fasta"):
        print(seq.id)
        #print(seq.seq)
        print(cleanup(seq.seq))
        exit(1)

    

def cleanup(sequence=""):

    sequence.strip("Nn")
    sequence = sequence.upper()

    return sequence


# def padding():

def main(filepath=""):
    
    print(f"Reading FASTA file: {filepath}")
    read(filepath=filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str,
                        help="Path to the input FASTA file")
    
    args = parser.parse_args()
    
    main(filepath=args.filepath)
