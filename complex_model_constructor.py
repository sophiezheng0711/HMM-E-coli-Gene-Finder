import numpy as np
import pandas as pd
import config


def gen_all_states(codons):
    states = []
    # 61 coding regions
    for codon in codons:
        states.append((codon, 's1'))
        states.append((codon, 's2'))
        states.append((codon, 's3'))
        states.append((codon, 'd1'))
        states.append((codon, 'd2'))
        states.append((codon, 'd3'))
        states.append((codon, 'd4'))
    # stop codons
    states.append(('stop', 1))
    states.append(('stop', 2))
    states.append(('stop', 3))
    states.append(('stop', 4))
    states.append(('stop', 5))
    states.append(('stop', 6))
    # long intergenic region
    states.append(('long', 'd0'))
    for i in range(44):
        states.append(('long', ('d' + str(i+1))))
        states.append(('long', ('s' + str(i+1))))
    # short intergenic region
    states.append(('short', 'd0'))
    for i in range(9):
        states.append(('short', ('d' + str(i+1))))
        states.append(('short', ('s' + str(i+1))))
    # start codons
    states.append(('start', 1))
    states.append(('start', 2))
    states.append(('start', 3))
    # overlap 1
    states.append(('over1', 't1'))
    states.append(('over1', 'ag2'))
    states.append(('over1', 'a3'))
    states.append(('over1', 'g4'))
    states.append(('over1', 't5'))
    states.append(('over1', 'g6'))
    # overlap 4
    states.append(('over4', 'n1'))
    states.append(('over4', 'n2'))
    states.append(('over4', 'ag3'))
    states.append(('over4', 't4'))
    states.append(('over4', 'g5'))
    states.append(('over4', 'ag6'))
    states.append(('over4', 'n7'))
    states.append(('over4', 'n8'))
    return states

def gen_trans_probs(states, p_gene, p_indel, codon_freq, stop_freq, p_over):
    tb = np.zeros((len(states), len(states)))
    codon_start_idx = 0
    stop_start_idx = codon_start_idx + (7 * 61)
    long_start_idx = stop_start_idx + 6
    short_start_idx = long_start_idx + (2 * 44) + 1
    start_start_idx = short_start_idx + (2 * 9) + 1
    over1_start_idx = start_start_idx + 3
    over4_start_idx = over1_start_idx + 6
    
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
        tb[d_i4][stop1] = 1 * (1 - p_gene) * (1 - p_over) * stop_freq[0]
        tb[d_i4][stop4] = 1 * (1 - p_gene) * (1 - p_over) * stop_freq[1]
        tb[d_i4][over1_start_idx] = 1 * (1 - p_gene) * p_over * 0.5
        tb[d_i4][over4_start_idx] = 1 * (1 - p_gene) * p_over * 0.5
        tb[s_i2][stop1] = (p_indel * 1) * (1 - p_gene) * (1 - p_over) * stop_freq[0]
        tb[s_i2][stop4] = (p_indel * 1) * (1 - p_gene) * (1 - p_over) * stop_freq[1]
        tb[s_i2][over1_start_idx] = 1 * (1 - p_gene) * p_over * 0.5
        tb[s_i2][over4_start_idx] = 1 * (1 - p_gene) * p_over * 0.5
        tb[s_i3][stop1] = ((1 - p_indel) * 1) * (1 - p_gene) * (1 - p_over) * stop_freq[0]
        tb[s_i3][stop4] = ((1 - p_indel) * 1) * (1 - p_gene) * (1 - p_over) * stop_freq[1]
        tb[s_i3][over1_start_idx] = 1 * (1 - p_gene) * p_over * 0.5
        tb[s_i3][over4_start_idx] = 1 * (1 - p_gene) * p_over * 0.5

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
    # last stop codons connect to long intergene
    d_0 = long_start_idx
    d_1 = d_0 + 1

    # long intergenic region

    # stop3, stop6, d0 all connect to d0
    tb[stop_start_idx+2][d_0] = 1/(44+9+2+1)
    tb[d_0-1][d_0] = 1/(44+9+2+1)
    tb[d_0][d_0] = 1/(44+2)
    # stop3, stop6, d0 all connect to start1
    tb[stop_start_idx+2][start_start_idx] = 1/(44+9+2+1)
    tb[d_0-1][start_start_idx] = 1/(44+9+2+1)
    tb[d_0][start_start_idx] = 1/(44+2)
    # stop3, stop6, d0 all connect to all s nodes
    for i in range(44):
        d_i = d_1 + (i * 2)
        s_i = d_i + 1
        tb[stop_start_idx+2][s_i] = 1/(44+9+2+1)
        tb[d_0-1][s_i] = 1/(44+9+2+1)
        tb[d_0][s_i] = 1/(44+2)
    # d_i, for all i, connects to d_i and s_j, for all j >= i + 1, as well as start1
    # s_i, for all i, connects to d_i and s_j, for all j >= i + 1, as well as start1
    for i in range(44):
        d_i = d_1 + (i * 2)
        tb[d_i][d_i] = 1/(44 - i + 1)
        tb[d_i][start_start_idx] = 1/(44 - i + 1)
        tb[d_i+1][d_i] = 1/(44 - i + 1)
        tb[d_i+1][start_start_idx] = 1/(44 - i + 1)
        j = i + 1
        while j < 44:
            s_j = d_1 + (j * 2) + 1
            tb[d_i][s_j] = 1/(44 - i + 1)
            tb[d_i+1][s_j] = 1/(44 - i + 1)
            j += 1

    # last stop codons connect to short intergene
    d_0 = short_start_idx
    d_1 = d_0 + 1

    # short intergenic region
    
    # stop3, stop6, d0 all connect to d0
    tb[stop_start_idx+2][d_0] = 1/(44+9+2+1)
    tb[long_start_idx-1][d_0] = 1/(44+9+2+1)
    tb[d_0][d_0] = 1/(9+2)
    # stop3, stop6, d0 all connect to start1
    tb[d_0][start_start_idx] = 1/(9+2)
    # stop3, stop6, d0 all connect to all s nodes
    for i in range(9):
        d_i = d_1 + (i * 2)
        s_i = d_i + 1
        tb[stop_start_idx+2][s_i] = 1/(44+9+2+1)
        tb[long_start_idx-1][s_i] = 1/(44+9+2+1)
        tb[d_0][s_i] = 1/(9+2)
    # d_i, for all i, connects to d_i and s_j, for all j >= i + 1, as well as start1
    # s_i, for all i, connects to d_i and s_j, for all j >= i + 1, as well as start1
    for i in range(9):
        d_i = d_1 + (i * 2)
        tb[d_i][d_i] = 1/(9 - i + 1)
        tb[d_i][start_start_idx] = 1/(9 - i + 1)
        tb[d_i+1][d_i] = 1/(9 - i + 1)
        tb[d_i+1][start_start_idx] = 1/(9 - i + 1)
        j = i + 1
        while j < 9:
            s_j = d_1 + (j * 2) + 1
            tb[d_i][s_j] = 1/(9 - i + 1)
            tb[d_i+1][s_j] = 1/(9 - i + 1)
            j += 1

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
    # same for the last nodes of the overlap models
    start3 = start_start_idx + 2
    stop1 = stop_start_idx
    stop4 = stop1 + 3
    over1_6 = over1_start_idx + 5
    over4_8 = over4_start_idx + 7
    for j in range(61):
        s_j1 = codon_start_idx + (j * 7)
        s_j2 = s_j1 + 1
        d_j1 = s_j1 + 3
        tb[start3][s_j1] = 1 * (p_gene * codon_freq[j]) * (1 - (2 * p_indel))
        tb[start3][s_j2] = 1 * (p_gene * codon_freq[j]) * p_indel
        tb[start3][d_j1] = 1 * (p_gene * codon_freq[j]) * p_indel
        tb[over1_6][s_j1] = 1 * (p_gene * codon_freq[j]) * (1 - (2 * p_indel))
        tb[over1_6][s_j2] = 1 * (p_gene * codon_freq[j]) * p_indel
        tb[over1_6][d_j1] = 1 * (p_gene * codon_freq[j]) * p_indel
        tb[over4_8][s_j1] = 1 * (p_gene * codon_freq[j]) * (1 - (2 * p_indel))
        tb[over4_8][s_j2] = 1 * (p_gene * codon_freq[j]) * p_indel
        tb[over4_8][d_j1] = 1 * (p_gene * codon_freq[j]) * p_indel
    tb[start3][stop1] = 1 * (1 - p_gene) * (1 - p_over) * stop_freq[0]
    tb[start3][stop4] = 1 * (1 - p_gene) * (1 - p_over) * stop_freq[1]
    tb[over1_6][stop1] = 1 * (1 - p_gene) * (1 - p_over) * stop_freq[0]
    tb[over1_6][stop4] = 1 * (1 - p_gene) * (1 - p_over) * stop_freq[1]
    tb[over4_8][stop1] = 1 * (1 - p_gene) * (1 - p_over) * stop_freq[0]
    tb[over4_8][stop4] = 1 * (1 - p_gene) * (1 - p_over) * stop_freq[1]
    tb[start3][over1_start_idx] = 1 * (1 - p_gene) * p_over * 0.5
    tb[start3][over4_start_idx] = 1 * (1 - p_gene) * p_over * 0.5
    tb[over1_6][over1_start_idx] = 1 * (1 - p_gene) * p_over * 0.5
    tb[over1_6][over4_start_idx] = 1 * (1 - p_gene) * p_over * 0.5
    tb[over4_8][over1_start_idx] = 1 * (1 - p_gene) * p_over * 0.5
    tb[over4_8][over4_start_idx] = 1 * (1 - p_gene) * p_over * 0.5

    # overlap model 1
    for i in range(5):
        o_i = over1_start_idx + i
        if i != 2:
            tb[o_i][o_i+1] = 1
    tb[over1_start_idx+1][over1_start_idx+3] = 1
    # overlap model 4
    for i in range(7):
        o_i = over4_start_idx + i
        tb[o_i][o_i+1] = 1
    return tb

def gen_emiss_probs(states, bases, codons, taa_tga):
    taa_tga /= np.sum(taa_tga)
    emiss_probs = np.zeros((len(states), len(bases)))
    bases_idx_map = {v:i for i,v in enumerate(bases)}
    emiss_map = {
        'stop': [[0,0,0,1],[taa_tga[0],taa_tga[1],0,0],[1,0,0,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]],
        'intergene': config.intergene,
        'start': [config.start1_emiss,[0,0,0,1],[0,1,0,0]],
        # TODO: change this
        'ag' : [0.5,0.5,0,0]
    }

    # TODO: change this to long and short intergenic regions emission probabilities
    dummy_distribution = [0.25, 0.25, 0.25, 0.25]

    codon_start_idx = 0
    stop_start_idx = codon_start_idx + (7 * 61)
    long_start_idx = stop_start_idx + 6
    short_start_idx = long_start_idx + (2 * 44) + 1
    start_start_idx = short_start_idx + (2 * 9) + 1
    over1_start_idx = start_start_idx + 3
    over4_start_idx = over1_start_idx + 6

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
    
    # long intergenic region
    d_0 = long_start_idx
    emiss_probs[d_0] = dummy_distribution
    for i in range(44):
        d_i = d_0 + 1 + (i * 2)
        s_i = d_i + 1
        emiss_probs[d_i] = dummy_distribution
        emiss_probs[s_i] = dummy_distribution

    # short intergenic region
    d_0 = short_start_idx
    emiss_probs[d_0] = dummy_distribution
    for i in range(9):
        d_i = d_0 + 1 + (i * 2)
        s_i = d_i + 1
        emiss_probs[d_i] = dummy_distribution
        emiss_probs[s_i] = dummy_distribution

    for i in range(3):
        emiss_probs[start_start_idx+i] = emiss_map['start'][i]

    # overlap model 1
    emiss_probs[over1_start_idx][bases_idx_map['T']] = 1
    emiss_probs[over1_start_idx+1] = emiss_map['ag']
    emiss_probs[over1_start_idx+2][bases_idx_map['A']] = 1
    emiss_probs[over1_start_idx+3][bases_idx_map['G']] = 1
    emiss_probs[over1_start_idx+4][bases_idx_map['T']] = 1
    emiss_probs[over1_start_idx+5][bases_idx_map['G']] = 1
    # overlap model 4
    emiss_probs[over4_start_idx] = dummy_distribution
    emiss_probs[over4_start_idx+1] = dummy_distribution
    emiss_probs[over4_start_idx+2] = emiss_map['ag']
    emiss_probs[over4_start_idx+3][bases_idx_map['T']] = 1
    emiss_probs[over4_start_idx+4][bases_idx_map['G']] = 1
    emiss_probs[over4_start_idx+5] = emiss_map['ag']
    emiss_probs[over4_start_idx+6] = dummy_distribution
    emiss_probs[over4_start_idx+7] = dummy_distribution
    return emiss_probs
