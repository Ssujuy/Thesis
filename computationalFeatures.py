START_CODON = "ATG"
STOP_CODON = {"TAA", "TAG", "TGA"}

def hexamerScore():

    return None

def firstORFLength(sequence: str) -> int:

    startCodon = sequence.find(START_CODON)

    if startCodon == -1:
        return 0
    
    else:
        for i in range(startCodon, len(sequence), 3):
            codon = sequence[i:i+3]
            if codon in STOP_CODON:
                return i + 3 - startCodon
        return len(sequence) - startCodon
    
def maxORFLength(sequence: str) -> int:

    maxLength = 0

    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3]

        if codon == START_CODON:
            for j in range(i+3, len(sequence) - 2, 3):
                codon2 = sequence[j:j+3]

                if codon2 in STOP_CODON:
                    length = j + 3 - i

                    if length > maxLength:
                        maxLength = length
                    break
            if maxLength == 0:
                maxLength = len(sequence) - i

    return maxLength