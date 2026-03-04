import os
import re
import sys
import random
import argparse
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import Helpers
import Types

stopCodons = {"TAA", "TAG", "TGA"}

def filtering(ncbiFasta, openProtFasta):
    """
    Filters sequence records from NCBI and OpenProt FASTA files based on specific markers.
    
    NCBI records are kept if their description contains "protein_id=NP_".
    OpenProt records are kept if their ID does not start with "IP_" and their description contains "NM_", "XM_", or "ENST".

    Parameters
    ----------
    ncbiFasta : str
        Filepath or handle to the input NCBI FASTA file.
    openProtFasta : str
        Filepath or handle to the input OpenProt FASTA file.

    Return
    ----------
    list
        List of Biopython SeqRecord objects that met the filtering criteria.
    """
    Helpers.colourPrint(Types.Colours.WHITE,"Filtering NCBI and OpenProt databases")
    ncbiKept = 0
    opKept = 0
    sequenceRecords = []

    for record in SeqIO.parse(ncbiFasta, "fasta"):
        if "protein_id=NP_" in record.description:
            sequenceRecords.append(record)
            ncbiKept += 1

    for record in SeqIO.parse(openProtFasta, "fasta"):
        if not record.id.startswith("IP_"):
            if any(m in record.description for m in ["NM_", "XM_", "ENST"]):
                sequenceRecords.append(record)
                opKept += 1

    Helpers.colourPrint(Types.Colours.GREEN,f"  -> NCBI Kept: {ncbiKept} | OpenProt Kept: {opKept} | Total: {len(sequenceRecords)}\n")
    return sequenceRecords


def extractSmorfs(inputRecords):
    """
    Filters out sequences longer than 303 characters to extract small 
    Open Reading Frames (smORFs) from the provided sequence records.

    Parameters
    ----------
    inputRecords : list
        List of sequence records (e.g., Biopython SeqRecord objects) to be filtered.

    Return
    ----------
    list
        A list containing the filtered sequence records with a length of 303 or less.
    """
    Helpers.colourPrint(Types.Colours.WHITE,"Extracting smORFs with length <= 303 from CD-HIT output")
    sequenceRecords = []
    
    for record in inputRecords:
        if len(record.seq) <= 303:
            sequenceRecords.append(record)
            
    Helpers.colourPrint(Types.Colours.GREEN,f"  -> Kept {len(sequenceRecords)} smORFs.\n")

    return sequenceRecords


def finalizeWithStrictCuration(inputRecords):
    """
    Applies strict biological curation rules and deduplicates sequence records by Gene ID.

    The curation process filters sequences based on the following structural criteria:
    - Length must be a multiple of 3 (in-frame).
    - Length must be strictly between 36 and 306 nucleotides.
    - The sequence must terminate with a valid stop codon.
    - The sequence must not contain any internal stop codons.
    
    It also filters by database source, retaining curated RefSeq (NM_/NP_) and Ensembl (ENSP) records while explicitly rejecting predicted models (XM_/XP_). 
    Finally, it deduplicates the remaining records so that only one sequence is kept per unique Gene ID.

    Parameters
    ----------
    inputRecords : list
        List of sequence records (e.g., Biopython SeqRecord objects) to be curated.

    Return
    ----------
    list
        A list of curated, deduplicated sequence records meeting all strict criteria.
    """
    Helpers.colourPrint(Types.Colours.WHITE,"Applying strict curation and deduplicating by Gene ID")
    finalRecords = []
    seenGenes = set()
    counts = {"curatedNmNp": 0, "ensemblEnsp": 0, "predictedXmXp": 0}

    for record in inputRecords:
        seqStr = str(record.seq).upper()
        descStr = record.description.upper()
        recordId = record.id.upper()

        if len(seqStr) % 3 != 0 or len(seqStr) < 36 or len(seqStr) > 306: continue
        if seqStr[-3:] not in stopCodons: continue
        if any(seqStr[3:-3][i:i+3] in stopCodons for i in range(0, len(seqStr[3:-3]), 3)): continue

        geneMatch = re.search(r'\[gene=(.*?)\]', record.description)
        if not geneMatch:
            geneMatch = re.search(r'gene_name=([^| \n]*)', record.description)
        geneId = geneMatch.group(1) if geneMatch else recordId.split('|')[0]

        isCuratedRefseq = "NP_" in recordId or "NM_" in recordId or "NP_" in descStr or "NM_" in descStr
        isEnsembl = "ENSP" in recordId or "ENSP" in descStr
        isPredicted = "XP_" in recordId or "XM_" in recordId or "XP_" in descStr or "XM_" in descStr

        if (isCuratedRefseq or isEnsembl) and not isPredicted:
            if geneId not in seenGenes:
                finalRecords.append(record)
                seenGenes.add(geneId)
                if isCuratedRefseq: counts["curatedNmNp"] += 1
                else: counts["ensemblEnsp"] += 1
        elif isPredicted:
            counts["predictedXmXp"] += 1

    Helpers.colourPrint(Types.Colours.GREEN,f"  Curated RefSeq:      {counts['curatedNmNp']}")
    Helpers.colourPrint(Types.Colours.GREEN,f"  Ensembl:             {counts['ensemblEnsp']}")
    Helpers.colourPrint(Types.Colours.GREEN,f"  Predicted (Ignored): {counts['predictedXmXp']}")
    Helpers.colourPrint(Types.Colours.GREEN,f"  -> Final Curated Selection: {len(finalRecords)}\n")
    return finalRecords


def buildMasterChrMap(genomeDict):
    """
    Builds a comprehensive mapping dictionary for human chromosomes, linking NCBI RefSeq accessions, 'chr'-prefixed names, and standard integer strings 
    to their corresponding GenBank (CM_) accession numbers.

    The function initializes with a base mapping of GRCh38 NC_ accessions to CM_ accessions.
    It then dynamically parses the NC_ strings to generate additional alias keys.
    For example, 'NC_000001.11' will also create keys for 'chr1' and '1', all mapping to 'CM000663.2'.
    This ensures robust and flexible chromosome identification during downstream processing.

    Parameters
    ----------
    genomeDict : dict
        A dictionary or mapping representing the genome reference.

    Return
    ----------
    dict
        A dictionary where keys are various chromosome identifiers 
        (e.g., 'NC_000001.11', 'chr1', '1') and values are the corresponding 
        GenBank CM_ accession strings.
    """
    mapping = {
        "NC_000001.11": "CM000663.2", "NC_000002.12": "CM000664.2", "NC_000003.12": "CM000665.2",
        "NC_000004.12": "CM000666.2", "NC_000005.10": "CM000667.2", "NC_000006.12": "CM000668.2",
        "NC_000007.14": "CM000669.2", "NC_000008.11": "CM000670.2", "NC_000009.12": "CM000671.2",
        "NC_000010.11": "CM000672.2", "NC_000011.10": "CM000673.2", "NC_000012.12": "CM000674.2",
        "NC_000013.11": "CM000675.2", "NC_000014.9": "CM000676.2",  "NC_000015.10": "CM000677.2",
        "NC_000016.10": "CM000678.2", "NC_000017.11": "CM000679.2", "NC_000018.10": "CM000680.2",
        "NC_000019.10": "CM000681.2", "NC_000020.11": "CM000682.2", "NC_000021.9": "CM000683.2",
        "NC_000022.11": "CM000684.2", "NC_000023.11": "CM000685.2", "NC_000024.10": "CM000686.2"
    }
    for ncKey, cmVal in mapping.copy().items():
        chrNum = str(int(ncKey.split('_')[1].split('.')[0]))
        mapping[f"chr{chrNum}"] = cmVal
        mapping[chrNum] = cmVal
    return mapping


def getTrueUpstreamClean(inputRecords, genomeFasta, bedFile):
    """
    Extracts the true 3-nucleotide upstream sequence for each record and prepends it to the coding sequence (CDS).

    This function attempts to find the genomic coordinates of each sequence first by parsing the FASTA description header (handling NCBI 'join' and 'complement' syntax).
    If the header lacks this information, it falls back to a coordinate lookup using the provided BED file. 
    
    Once coordinates are found, it extracts the 3 base pairs immediately upstream of the sequence from the reference genome.
    Finally, it truncates the original sequence to start at the first "ATG" (start codon) and prepends the true upstream sequence to it.

    Parameters
    ----------
    inputRecords : list
        List of Biopython SeqRecord objects to process.
    genomeFasta : str
        Filepath to the reference genome FASTA file.
    bedFile : str
        Filepath to the BED file containing genomic coordinates. Expected to 
        have standard BED columns (chrom, start, end, name, strand).

    Return
    ----------
    list
        A list of modified SeqRecord objects that successfully obtained a valid 
        3-bp upstream sequence (without 'N's).
    """
    Helpers.colourPrint(Types.Colours.WHITE,"Loading Genome and BED to extract True Upstream Sequence")
    genomeDict = SeqIO.to_dict(SeqIO.parse(genomeFasta, "fasta"))
    chrMap = buildMasterChrMap(genomeDict)
    
    bedData = pd.read_csv(bedFile, sep='\t', header=None, usecols=[0, 1, 2, 3, 5], 
                          names=['chrom', 'start', 'end', 'name', 'strand'])
    bedLookup = {str(n): (c, s, e, st) for c, s, e, n, st in bedData.itertuples(index=False)}

    finalRecords = []
    for record in inputRecords:
        upCodon = "NNN"
        descStr = record.description
        
        locMatch = re.search(r'location=(?:complement\()?join\(([\d\.,]+)\)|location=(?:complement\()?([\d\.\,]+)', descStr)
        accMatch = re.search(r'NC_\d+\.\d+', descStr)
        
        if accMatch and locMatch:
            cmAcc = chrMap.get(accMatch.group(0))
            isComp = 'complement' in descStr
            ptsList = [int(x) for x in re.findall(r'\d+', locMatch.group(0))]
            if cmAcc in genomeDict:
                seqRef = genomeDict[cmAcc].seq
                if not isComp:
                    startPt = ptsList[0] - 1
                    upCodon = str(seqRef[startPt-3:startPt]).upper()
                else:
                    endPt = ptsList[-1]
                    upCodon = str(seqRef[endPt:endPt+3].reverse_complement()).upper()

        if upCodon == "NNN":
            cleanId = record.id.split('|')[0]
            if cleanId in bedLookup:
                c, s, e, st = bedLookup[cleanId]
                cmAcc = chrMap.get(str(c))
                if cmAcc in genomeDict:
                    seqRef = genomeDict[cmAcc].seq
                    if st == '+':
                        upCodon = str(seqRef[s-3:s]).upper()
                    else:
                        upCodon = str(seqRef[e:e+3].reverse_complement()).upper()

        if len(upCodon) == 3 and "N" not in upCodon:
            seqStr = str(record.seq).upper()
            atgIdx = seqStr.find("ATG")
            cdsStr = seqStr[atgIdx:] if atgIdx != -1 else seqStr
            record.seq = Seq(upCodon + cdsStr)
            finalRecords.append(record)

    Helpers.colourPrint(Types.Colours.GREEN,f"  -> Final Clean Positive Count: {len(finalRecords)}\n")
    return finalRecords


def findMaxOrf(sequenceObj):
    """
    Scans a nucleotide sequence to find the longest Open Reading Frame (ORF).

    The function searches the sequence for a canonical start codon ("ATG"). 
    Once found, it scans downstream in-frame (steps of 3 nucleotides) until it encounters a valid stop codon.
    It evaluates all possible ORFs across all forward reading frames and returns the longest one.

    Parameters
    ----------
    sequenceObj : str or Seq
        The nucleotide sequence to be analyzed. Can be a standard string 
        or a Biopython Seq object.

    Return
    ----------
    tuple
        A tuple containing two elements:
        - maxOrf (str): The longest extracted ORF sequence (from "ATG" to 
          the stop codon). Returns an empty string if no ORF is found.
        - maxStartIndex (int): The 0-based index where this longest ORF begins. 
          Returns -1 if no ORF is found.
    """
    seqStr = str(sequenceObj).upper()
    maxOrf = ""
    maxStartIndex = -1
    
    for i in range(len(seqStr)):
        if seqStr[i:i+3] == "ATG":
            for j in range(i + 3, len(seqStr), 3):
                currentCodon = seqStr[j:j+3]
                if currentCodon in stopCodons:
                    currentOrf = seqStr[i:j+3]
                    if len(currentOrf) > len(maxOrf):
                        maxOrf = currentOrf
                        maxStartIndex = i
                    break
    return maxOrf, maxStartIndex


def buildNegativeSet(inputFasta, targetCount):
    """
    Constructs a balanced dataset of non-coding small Open Reading Frames (smORFs) to serve as negative training samples.

    The function scans sequences from a provided FASTA file to find their longest Open Reading Frames (ORFs).
    It filters these ORFs strictly by length (between 33 and 303 nucleotides).
    For valid ORFs, it attempts to extract the 3 base pairs immediately upstream.
    If the upstream sequence is incomplete or contains ambiguous bases ('N'), the sequence is discarded.
    Finally, it randomly samples the valid candidates to match the requested target count, ensuring class balance.

    Parameters
    ----------
    inputFasta : str
        Filepath to the input FASTA file containing non-coding RNA sequences.
    targetCount : int
        The desired number of negative samples to randomly extract.

    Return
    ----------
    list
        A list of Biopython SeqRecord objects containing the prepared negative 
        smORF sequences (3-bp upstream sequence + the smORF).
    """
    Helpers.colourPrint(Types.Colours.WHITE,f"Building non-coding smORF dataset {inputFasta}")
    allValidNegatives = []
    
    for record in SeqIO.parse(inputFasta, "fasta"):
        orfSeq, startIdx = findMaxOrf(record.seq)
        
        if len(orfSeq) < 33 or len(orfSeq) > 303:
            continue
            
        if startIdx >= 3:
            upstreamStr = str(record.seq[startIdx-3:startIdx]).upper()
        else:
            upstreamStr = "N" * (3 - startIdx) + str(record.seq[0:startIdx]).upper()
            
        if len(upstreamStr) == 3 and "N" not in upstreamStr:
            fullNegativeSeq = upstreamStr + orfSeq
            record.seq = Seq(fullNegativeSeq)
            allValidNegatives.append(record)

    Helpers.colourPrint(Types.Colours.PURPLE,f"  -> Total potential ncRNA ORFs found: {len(allValidNegatives)}")

    if len(allValidNegatives) >= targetCount:
        finalNegatives = random.sample(allValidNegatives, targetCount)
    else:
        Helpers.colourPrint(Types.Colours.RED,"  -> Warning: Not enough ncRNA ORFs to reach target count! Using all available.")
        finalNegatives = allValidNegatives

    Helpers.colourPrint(Types.Colours.GREEN,f"  -> Successfully prepared {len(finalNegatives)} balanced negative samples.\n")
    return finalNegatives


def main():
    parser = argparse.ArgumentParser(description="End-to-End Data Extraction Pipeline")
    parser.add_argument("--ncbiFasta", default="GCF_000001405.40_GRCh38.p14_cds_from_genomic.fna", help="Path to NCBI FASTA file")
    parser.add_argument("--openProtFasta", default="human-openprot-r1_6-refprots+altprots+isoforms_min_2_pep-.dna.fasta", help="Path to OpenProt FASTA file")
    parser.add_argument("--openProtBed", default="human-openprot-r1_6-refprots+altprots+isoforms_min_2_pep-.bed", help="Path to OpenProt BED file")
    parser.add_argument("--genomeFasta", default="GCA_000001405.29_GRCh38.p14_genomic.fna", help="Path to the reference Genome FASTA file")
    parser.add_argument("--ncRnaFasta", default="Homo_sapiens.GRCh38.ncrna.fa", help="Path to the ncRNA Ensembl FASTA for negative generation")
    
    parser.add_argument("--outStep1", default="data/extracted/socp_step1_integrated_no_alts.fasta", help="Output file for Step 1 (pre-CD-HIT)")
    parser.add_argument("--inCdhit", default="data/extracted/socp_step2_nr_no_alts.fasta", help="Expected input file from CD-HIT for Step 3")
    
    parser.add_argument("--outPositive", default="data/extracted/coding_smorfs_2pep_5890.fa", help="Output path for Positive Dataset")
    parser.add_argument("--outNegative", default="data/extracted/non_coding_smorfs_2pep_5890.fa", help="Output path for Negative Dataset")
    
    args = parser.parse_args()

    if not os.path.exists(args.inCdhit):

        Helpers.colourPrint(Types.Colours.BLUE,"=== Starting Data Extraction Pipeline ===\n")

        records = filtering(args.ncbiFasta, args.openProtFasta)

        SeqIO.write(records, args.outStep1, "fasta")
        Helpers.colourPrint(Types.Colours.GREEN,f"Saved pre-clustering sequences to: {args.outStep1}\n")

        Helpers.colourPrint(Types.Colours.RED,f"Could not find the expected CD-HIT output file: '{args.inCdhit}'")
        Helpers.colourPrint(Types.Colours.WHITE,f"\Run CD-HIT command in the terminal: cd-hit -i {args.outStep1} -o {args.inCdhit} -c 0.9 -s 0.9 -T 8 -M 8000")
        Helpers.colourPrint(Types.Colours.WHITE,f"Once CD-HIT finishes and produces '{args.inCdhit}', re-run this script with the exact same arguments.\n")
        sys.exit(0)

    Helpers.colourPrint(Types.Colours.PURPLE,f"Found CD-HIT output '{args.inCdhit}'. Resuming pipeline...\n")

    records = list(SeqIO.parse(args.inCdhit, "fasta"))

    records = extractSmorfs(records)

    records = finalizeWithStrictCuration(records)

    records = getTrueUpstreamClean(records, args.genomeFasta, args.openProtBed)

    SeqIO.write(records, args.outPositive, "fasta")
    Helpers.colourPrint(Types.Colours.GREEN,f"Saved Positive Dataset to: {args.outPositive}")

    targetCount = len(records)

    negativeRecords = buildNegativeSet(args.ncRnaFasta, targetCount)

    SeqIO.write(negativeRecords, args.outNegative, "fasta")
    Helpers.colourPrint(Types.Colours.GREEN,f"Saved Negative Dataset to: {args.outNegative}")

    Helpers.colourPrint(Types.Colours.BLUE,"\nCleaning up temporary clustering files...")
    tempFilesToClean = [args.outStep1, args.inCdhit, args.inCdhit + ".clstr"]
    for tempFile in tempFilesToClean:
        if os.path.exists(tempFile):
            os.remove(tempFile)
            Helpers.colourPrint(Types.Colours.RED,f"  -> Deleted {tempFile}")

    Helpers.colourPrint(Types.Colours.BLUE,"\n=== Pipeline Execution Complete ===")


if __name__ == "__main__":
    main()