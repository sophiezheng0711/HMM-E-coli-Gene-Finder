import numpy as np
import matplotlib.pyplot as plt
from complex_model_constructor import gen_trans_probs, gen_all_states, gen_emiss_probs
import pandas as pd
import config
from Bio import SeqIO


def get_probabilities(p_over):
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
    trans_probs = gen_trans_probs(states, config.p_gene, config.p_indel, codon_freq, stop_freq, p_over)
    emiss_probs = gen_emiss_probs(states, bases, codons, taa_tga)
    trans_probs = np.log(trans_probs)
    emiss_probs = np.log(emiss_probs)
    init_probs = np.ones(len(states))
    init_probs /= np.sum(init_probs)
    init_probs = np.log(init_probs)

    return states, trans_probs, emiss_probs, init_probs

def sumLogProb(a, b):
    return np.logaddexp(a, b)

def forward_backward(obs, states, bases, trans_probs, emiss_probs, init_probs):
    base_to_idx = {v:i for i,v in enumerate(bases)}
    F = np.zeros((len(states), len(obs)))
    B = np.zeros((len(states), len(obs)))
    for i in range(len(states)):
        F[i][0] = init_probs[i] + emiss_probs[i][base_to_idx[obs[0]]]
    for i in range(1, len(obs)):
        for l in range(len(states)):
            F[l][i] = F[0][i-1] + trans_probs[0][l]
            for k in range(1, len(states)):
                F[l][i] = sumLogProb(F[l][i], F[k][i-1] + trans_probs[k][l])
            F[l][i] += emiss_probs[l][base_to_idx[obs[i]]]
    likelihood_f = F[0][len(obs)-1]
    for i in range(1, len(states)):
        likelihood_f = sumLogProb(likelihood_f, F[i][len(obs)-1])
    for i in range(len(states)):
        B[i][len(obs)-1] = 0
    for i in reversed(range(len(obs)-1)):
        for l in range(len(states)):
            B[l][i] = B[0][i+1] + trans_probs[0][l] + emiss_probs[0][base_to_idx[obs[i+1]]]
            for k in range(1, len(states)):
                B[l][i] = sumLogProb(B[l][i], B[k][i+1] + trans_probs[k][l] + emiss_probs[k][base_to_idx[obs[i+1]]])

    return (F, likelihood_f, B)


def em(states, bases, fb_output, obs, tp, ep):
    print('Starting EM...')
    F, likelihood, B = fb_output
    base_to_idx = {v:i for i,v in enumerate(bases)}
    E = np.zeros((len(states), len(bases)))
    A = np.zeros((len(states), len(states)))

    for k in range(len(states)):
        for l in range(len(states)):
            for i in range(len(obs) - 1):
                temp = F[k][i] + tp[k][l] + ep[l][base_to_idx[obs[i+1]]] + B[l][i+1]
                if A[k][l] == 0:
                    A[k][l] = temp
                else:
                    A[k][l] = sumLogProb(A[k][l], temp)
            A[k][l] -= likelihood

    for k in range(len(states)):
        for b in range(len(bases)):
            for i in range(len(obs)):
                if obs[i] == bases[b]:
                    temp = F[k][i] + B[k][i]
                    if E[k][b] == 0:
                        E[k][b] = temp
                    else:
                        E[k][b] = sumLogProb(E[k][b], temp)
            E[k][b] -= likelihood

    return A, E


def train(seq_lst, bases):
    states, trans_probs, emiss_probs, init_probs = get_probabilities(config.p_over)
    epochs = 1
    for epoch in range(epochs):
        for seq_idx in range(len(seq_lst)):
            seq = seq_lst[seq_idx]
            fb_output = forward_backward(seq, states, bases, trans_probs, emiss_probs, init_probs)
            trans_probs, emiss_probs = em(states, bases, fb_output, seq, trans_probs, emiss_probs)
            print("Sequence [%d/%d] in Epoch [%d/%d]" % (seq_idx+1, len(seq_lst), epoch+1, epochs))
        print()
    return (trans_probs, emiss_probs)


def main():
    fasta_sequences = list(SeqIO.parse(open('data/WIS_MG1655_v3.fas'),'fasta'))
    seq = str(fasta_sequences[0].seq)[:10000]
    seq_lst = [seq[i*1000:(i+1)*1000] for i in range(int(len(seq) / 1000))]
    bases = ['a', 'g', 'c', 't']
    trans_probs, emiss_probs = train(seq_lst, bases)
    with open('preprocessed/complex_trans_probs.npy', 'wb') as f:
        np.save(f, trans_probs)
    with open('preprocessed/complex_emiss_probs.npy', 'wb') as f:
        np.save(f, emiss_probs)

if __name__ == '__main__':
    main()
