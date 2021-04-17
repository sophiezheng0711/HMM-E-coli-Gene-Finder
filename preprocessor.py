import pandas as pd
from collections import defaultdict
from Bio import SeqIO
from Bio.Seq import Seq

# Getting the entire ecoli sequence from text data
fasta_sequences = list(SeqIO.parse(open('data/WIS_MG1655_v3.fas'),'fasta'))
ecoli_sequence = str(fasta_sequences[0].seq)

# Getting annotated ecoli sequence data 
df = pd.read_csv('data/WIS_MG1655_v3_features_coordinates.txt',sep='\t')
df = df[df['FeatureType'] == 'CDS']

# TODO: find init probs for each state

def find_intergenic_probs():
    """
    calculates the intergenic probabilities of each amino acid and p_interloop
    """
    intergenic_inds = set([i for i in range(len(ecoli_sequence))])
    for _, row in df.iterrows():
        intergenic_inds = intergenic_inds - set([i for i in range(row['Left End']-1, row['Right End'])])

    print(len(intergenic_inds) / len(ecoli_sequence))

    counts = defaultdict(int)
    for i in intergenic_inds:
        c = ecoli_sequence[i].lower()
        counts[c] += 1

    total = sum(counts.values())
    probs = []
    for _, v in counts.items():
        probs.append(v/total)

    intergenic_probs = pd.DataFrame({'probs' : probs}, index=counts.keys())
    intergenic_probs.to_csv('preprocessed/intergenic_probs.csv')

def find_codon_percentages():
    """
    calculates the start and stop codon percentages
    """
    start_codon_counts = defaultdict(int)
    end_codon_counts = defaultdict(int)
    for _, row in df.iterrows():
        start_ind = row['Left End']-1            
        end_ind = row['Right End']-1
        start_codon = ecoli_sequence[start_ind:start_ind+3].lower()
        end_codon = ecoli_sequence[end_ind-2:end_ind+1].lower()
        start_codon_counts[start_codon] += 1
        end_codon_counts[end_codon] += 1

    start_codons = ['atg', 'gtg', 'ttg']
    stop_codons = ['tag', 'taa', 'tga']
    start_sum = start_codon_counts['atg'] + start_codon_counts['gtg'] + start_codon_counts['ttg']
    end_sum = end_codon_counts['tag'] + end_codon_counts['taa'] + end_codon_counts['tga']
    start_probs = []
    end_probs = []

    for v in start_codons:
        start_probs.append(start_codon_counts[v]/start_sum)
    start_probs_df = pd.DataFrame({'start codon probs' : start_probs}, index=start_codons)
    start_probs_df.to_csv('preprocessed/start_codon_probs.csv')

    for v in stop_codons:
        end_probs.append(end_codon_counts[v]/end_sum)
    end_probs_df = pd.DataFrame({'stop codon probs' : end_probs}, index=stop_codons)
    end_probs_df.to_csv('preprocessed/stop_codon_probs.csv')

def find_average_codons():
    """
    Calculates the average number of codons in a gene
    """
    total = 0
    for _, row in df.iterrows():
        amino_acids = row['Right End'] - row['Left End'] + 1
        total += amino_acids/3
    return total / len(df)

def count_codons():
    """
    Finds the frequencies/counts of each codon
    """
    codons = ['AAA', 'AAG', 'AAC', 'AAT', 'AGA', 'AGG', 'AGC', 'AGT', 'ACA', 'ACG', 'ACC', 'ACT', 'ATA', 'ATG', 'ATC', 'ATT', 'GAA', 'GAG', 'GAC', 'GAT', 'GGA', 'GGG', 'GGC', 'GGT', 'GCA', 'GCG', 'GCC', 'GCT', 'GTA', 'GTG', 'GTC', 'GTT', 'CAA', 'CAG', 'CAC', 'CAT', 'CGA', 'CGG', 'CGC', 'CGT', 'CCA', 'CCG', 'CCC', 'CCT', 'CTA', 'CTG', 'CTC', 'CTT', 'TAC', 'TAT', 'TGG', 'TGC', 'TGT', 'TCA', 'TCG', 'TCC', 'TCT', 'TTA', 'TTG', 'TTC', 'TTT']
    d = defaultdict(int)
    for c in codons:
        codon = c.lower()
        for _, row in df.iterrows():
            start = row['Left End']-1
            end = row['Right End']
            gene = ecoli_sequence[start:end].lower()
            d[codon] += Seq(gene).count(codon)
    total = sum(d.values())
    probs = []
    for _, v in d.items():
        probs.append(v/total)
    codon_count_df = pd.DataFrame({'codon probs' : probs}, index=d.keys())
    codon_count_df.to_csv('preprocessed/codon_probs.csv')

count_codons()