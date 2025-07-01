START_CODON = "ATG"
STOP_CODON = {"TAA", "TAG", "TGA"}

def hexamerScore():

    return None

def orfLength(sequence: str) -> int:

    startCodon = sequence.find(START_CODON)

    if startCodon == -1:
        return 0
    
    else:
        for i in range(startCodon, len(sequence), 3):
            codon = sequence[i:i+3]
            if codon in STOP_CODON:
                return i + 3 - startCodon
        return len(sequence) - startCodon