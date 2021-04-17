import numpy as np
import pandas as pd
import config


def gen_all_states(codons):
    states = []
    for codon in codons:
        states.append((codon, 's1'))
        states.append((codon, 's2'))
        states.append((codon, 's3'))
        states.append((codon, 'd1'))
        states.append((codon, 'd2'))
        states.append((codon, 'd3'))
        states.append((codon, 'd4'))
    states.append(('stop', 1))
    states.append(('stop', 2))
    states.append(('stop', 3))
    states.append(('stop', 4))
    states.append(('stop', 5))
    states.append(('stop', 6))
    states.append((None, 'intergene'))
    states.append(('start', 1))
    states.append(('start', 2))
    states.append(('start', 3))
    return states

def gen_trans_probs(states, p_gene, p_indel, p_interloop, codon_freq, stop_freq):
    tb = np.zeros((len(states), len(states)))
    codon_start_idx = 0
    stop_start_idx = codon_start_idx + (7 * 61)
    intergene_index = stop_start_idx + 6
    start_start_idx = intergene_index + 1
    
    # codon sub HMM
    for i in range(61):
        square_idx = codon_start_idx + (i * 7)
        diamond_idx = square_idx + 3
        # first base, and second base connect to corresponding diamonds
        for j in range(2):
            s = square_idx + j
            tb[s][diamond_idx + j + 1] = p_indel
            tb[s][s + 1] = 1 - (2 * p_indel)
        # connect third base to diamond
        s = square_idx + 2
        tb[s][diamond_idx + 3] = p_indel
        # connect first base to third base
        tb[square_idx][square_idx + 2] = p_indel
        # first 3 diamonds connect to square + 1
        for j in range(3):
            d = diamond_idx + j
            tb[d][square_idx + j + 1] = 1
        
        # process nodes leading into central, which we deleted
        d_i4 = diamond_idx + 3
        s_i2 = square_idx + 1
        s_i3 = s_i2 + 1
        stop1 = stop_start_idx
        stop4 = stop1 + 3
        tb[d_i4][stop1] = 1 * (1 - p_gene) * stop_freq[0]
        tb[d_i4][stop4] = 1 * (1 - p_gene) * stop_freq[1]
        tb[s_i2][stop1] = (p_indel * 1) * (1 - p_gene) * stop_freq[0]
        tb[s_i2][stop4] = (p_indel * 1) * (1 - p_gene) * stop_freq[1]
        tb[s_i3][stop1] = ((1 - p_indel) * 1) * (1 - p_gene) * stop_freq[0]
        tb[s_i3][stop4] = ((1 - p_indel) * 1) * (1 - p_gene) * stop_freq[1]

        for j in range(61):
            s_j1 = codon_start_idx + (j * 7)
            s_j2 = s_j1 + 1
            d_j1 = s_j1 + 3
            tb[d_i4][s_j1] = 1 * (p_gene * codon_freq[j]) * (1 - (2 * p_indel))
            tb[d_i4][s_j2] = 1 * (p_gene * codon_freq[j]) * p_indel
            tb[d_i4][d_j1] = 1 * (p_gene * codon_freq[j]) * p_indel
            tb[s_i2][s_j1] = (p_indel * 1) * (p_gene * codon_freq[j]) * (1 - (2 * p_indel))
            tb[s_i2][s_j2] = (p_indel * 1) * (p_gene * codon_freq[j]) * p_indel
            tb[s_i2][d_j1] = (p_indel * 1) * (p_gene * codon_freq[j]) * p_indel
            tb[s_i3][s_j1] = ((1 - p_indel) * 1) * (p_gene * codon_freq[j]) * (1 - (2 * p_indel))
            tb[s_i3][s_j2] = ((1 - p_indel) * 1) * (p_gene * codon_freq[j]) * p_indel
            tb[s_i3][d_j1] = ((1 - p_indel) * 1) * (p_gene * codon_freq[j]) * p_indel

    """
        stop codons connect to each other:
        stop1 stop2 stop3
            O - O - O
            |       |
                    - O intergene
            |       |
            O - O - O
        stop4 stop5 stop6
    """
    for i in range(2):
        tb[stop_start_idx+i][stop_start_idx+i+1] = 1
        tb[stop_start_idx+i+1][stop_start_idx+i+2] = 1
        tb[stop_start_idx+i+3][stop_start_idx+i+4] = 1
        tb[stop_start_idx+i+4][stop_start_idx+i+5] = 1
    # last stop codons connect to intergene
    tb[stop_start_idx+2][intergene_index] = 1
    tb[intergene_index-1][intergene_index] = 1
    # intergene connects to itself, or start codon
    tb[intergene_index][intergene_index] = p_interloop
    tb[intergene_index][start_start_idx] = 1 - p_interloop
    """
        start codons connect to each other:
        stop1 stop2 stop3
        intergene   1   2   3
                O - O - O - O
        stop4 stop5 stop6
    """
    tb[start_start_idx][start_start_idx+1] = 1
    tb[start_start_idx+1][start_start_idx+2] = 1
    # last start codon connects to central, which means it connects to all the nodes central connects to
    start3 = start_start_idx + 2
    stop1 = stop_start_idx
    stop4 = stop1 + 3
    for j in range(61):
        s_j1 = codon_start_idx + (j * 7)
        s_j2 = s_j1 + 1
        d_j1 = s_j1 + 3
        tb[start3][s_j1] = 1 * (p_gene * codon_freq[j]) * (1 - (2 * p_indel))
        tb[start3][s_j2] = 1 * (p_gene * codon_freq[j]) * p_indel
        tb[start3][d_j1] = 1 * (p_gene * codon_freq[j]) * p_indel
    tb[start3][stop1] = 1 * (1 - p_gene) * stop_freq[0]
    tb[start3][stop4] = 1 * (1 - p_gene) * stop_freq[1]
    return tb

def gen_emiss_probs(states, bases, codons, taa_tga):
    taa_tga /= np.sum(taa_tga)
    emiss_probs = np.zeros((len(states), len(bases)))
    bases_idx_map = {v:i for i,v in enumerate(bases)}
    emiss_map = {
        'stop': [[0,0,0,1],[taa_tga[0],taa_tga[1],0,0],[1,0,0,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]],
        'intergene': config.intergene,
        'start': [config.start1_emiss,[0,0,0,1],[0,1,0,0]]
    }

    codon_start_idx = 0
    stop_start_idx = codon_start_idx + (7 * 61)
    intergene_index = stop_start_idx + 6
    start_start_idx = intergene_index + 1

    for i in range(61):
        codon = codons[i]
        square_idx = codon_start_idx + (i * 7)
        diamond_idx = square_idx + 3
        for j in range(3):
            emiss_probs[square_idx+j][bases_idx_map[codon[j]]] = 1
        for j in range(4):
            emiss_probs[diamond_idx+j] = emiss_map['intergene']
    
    for i in range(6):
        emiss_probs[stop_start_idx+i] = emiss_map['stop'][i]
    emiss_probs[intergene_index] = emiss_map['intergene']
    for i in range(3):
        emiss_probs[start_start_idx+i] = emiss_map['start'][i]
    return emiss_probs

if __name__ == "__main__":
    bases = ['A', 'G', 'C', 'T']
    
    codon_frequencies_raw = pd.read_csv('preprocessed/codon_frequencies.csv', sep=',', header=None)
    # the 61 codons
    codons = codon_frequencies_raw[0].to_numpy()
    # their frequencies from page 3 table, should add up to approximately 1
    codon_freq = codon_frequencies_raw[1].to_numpy()
    codon_freq /= np.sum(codon_freq)

    taa_tga = np.array(config.taa_tga)
    stop_freq = [np.sum(taa_tga), 1 - np.sum(taa_tga)]

    states = gen_all_states(codons)
    trans_probs = gen_trans_probs(states, config.p_gene, config.p_indel, config.p_interloop, codon_freq, stop_freq)
    emiss_probs = gen_emiss_probs(states, bases, codons, taa_tga)
    trans_probs = np.log(trans_probs)
    emiss_probs = np.log(emiss_probs)

    with open('preprocessed/trans_probs.npy', 'wb') as f:
        np.save(f, trans_probs)
    with open('preprocessed/emiss_probs.npy', 'wb') as f:
        np.save(f, emiss_probs)
