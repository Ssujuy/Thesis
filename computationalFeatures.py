START_CODON = "ATG"
STOP_CODON = ""

def hexamerScore():

    return None

def orfLength(sequence: str) -> int:

    startCodon = sequence.find(START_CODON)

    if startCodon == -1:
        return 0
    
    else:
        for i in range(startCodon, len(sequence), 3):
            codon = sequence[i:i+3]
            if codon in {"TAA", "TAG", "TGA"}:
                return i + 3 - startCodon
        return len(sequence) - startCodon