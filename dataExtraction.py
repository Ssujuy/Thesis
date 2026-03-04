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
    Helpers.colourPrint(Types.Colours.WHITE,"Extracting smORFs with length <= 303 from CD-HIT output")
    sequenceRecords = []
    
    for record in inputRecords:
        if len(record.seq) <= 303:
            sequenceRecords.append(record)
            
    Helpers.colourPrint(Types.Colours.GREEN,f"  -> Kept {len(sequenceRecords)} smORFs.\n")

    return sequenceRecords


def finalizeWithStrictCuration(inputRecords):
    print()
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
    parser = argparse.ArgumentParser(description="End-to-End Thesis Data Extraction Pipeline")
    parser.add_argument("--ncbiFasta", required=True, help="Path to NCBI FASTA file")
    parser.add_argument("--openProtFasta", required=True, help="Path to OpenProt FASTA file")
    parser.add_argument("--openProtBed", required=True, help="Path to OpenProt BED file")
    parser.add_argument("--genomeFasta", required=True, help="Path to the reference Genome FASTA file")
    parser.add_argument("--ncRnaFasta", required=True, help="Path to the ncRNA Ensembl FASTA for negative generation")
    
    parser.add_argument("--outStep1", default="socp_step1_integrated_no_alts.fasta", help="Output file for Step 1 (pre-CD-HIT)")
    parser.add_argument("--inCdhit", default="socp_step2_nr_no_alts.fasta", help="Expected input file from CD-HIT for Step 3")
    
    parser.add_argument("--outPositive", default="coding_smorfs_2pep_5890.fa", help="Output path for Positive Dataset")
    parser.add_argument("--outNegative", default="non_coding_smorfs_2pep_5890.fa", help="Output path for Negative Dataset")
    
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