import numpy as np
import pandas as pd
from simple_model_constructor import gen_all_states as gen_simple_states
from complex_model_constructor import gen_all_states as gen_complex_states
from Bio import SeqIO
import argparse

def viterbi(obs, trans_probs, emiss_probs, init_probs, states, bases):
    dp = np.zeros((len(states), len(obs)))
    t = np.zeros((len(states), len(obs)))
    base_idx_map = {v:i for i,v in enumerate(bases)}
    for i in range(len(states)):
        dp[i][0] = init_probs[i] + emiss_probs[i][base_idx_map[obs[0]]]
        t[i][0] = i
    for i in range(1, len(obs)):
        for l in range(len(states)):
            max_value = np.NINF
            max_ind = -1
            for k in range(len(states)):
                max_value = max(max_value, dp[k][i-1] + trans_probs[k][l])
                if max_value == dp[k][i-1] + trans_probs[k][l]:
                    max_ind = k
            dp[l][i] = max_value + emiss_probs[l][base_idx_map[obs[i]]]
            t[l][i] = max_ind
        # if (i % 100 == 0):
        #     print("[" + str(i) + "/" + str(len(obs)-1) + "]")
    l = np.zeros(len(obs))
    i = len(obs) - 1
    l[i] = np.argmax(dp[:, i])
    while i >= 1:
        l[i-1] = t[int(l[i])][i]
        i -= 1
    ll = []
    for i in l:
        ll.append(states[int(i)])
    return ll

def find_gene_intervals(path, buffer=0):
    intervals = []
    start = None
    for (i, (name, state)) in enumerate(path):
        if start != None:
            if name == 'stop' and (state == 3 or state == 6):
                intervals.append((start+1+buffer, i+1+buffer))
                start = None
        else:
            if name == 'start' and state == 1:
                start = i
            elif name == 'stop' and (state == 3 or state == 6) and len(intervals) == 0:
                intervals.append((None, i+1+buffer))
    if start != None:
        intervals.append((start+1+buffer, None))
    return intervals

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '-model', required=True)
    args = parser.parse_args()

    if args.m is not None:
        np_load_old = np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        trans_probs = None
        emiss_probs = None
        states = None

        codon_frequencies_raw = pd.read_csv('preprocessed/codon_frequencies.csv', sep=',', header=None)
        # the 61 codons
        codons = codon_frequencies_raw[0].to_numpy()

        if args.m == 'simple':
            with open('preprocessed/trans_probs.npy', 'rb') as f:
                trans_probs = np.load(f)
            with open('preprocessed/emiss_probs.npy', 'rb') as f:
                emiss_probs = np.load(f)
            states = gen_simple_states(codons)
        elif args.m == 'complex':
            with open('preprocessed/complex_trans_probs.npy', 'rb') as f:
                trans_probs = np.load(f)
            with open('preprocessed/complex_emiss_probs.npy', 'rb') as f:
                emiss_probs = np.load(f)
            states = gen_complex_states(codons)
        else:
            raise ValueError

        init_probs = np.ones(len(states))
        init_probs /= np.sum(init_probs)
        init_probs = np.log(init_probs)

        fasta_sequences = list(SeqIO.parse(open('data/WIS_MG1655_v3.fas'),'fasta'))
        seq = str(fasta_sequences[0].seq)[:20000]
        seq_lst = [seq[i*1000:(i+1)*1000] for i in range(int(len(seq) / 1000))]
        intervals = []
        for i in range(len(seq_lst)):
            s = seq_lst[i]
            path = viterbi(s, trans_probs, emiss_probs, init_probs, states, ['a','g','c','t'])
            # print(path)
            intervals.extend(find_gene_intervals(path, i*1000))
            print("Epoch [%d/%d]" % (i+1, len(seq_lst)))
        print(intervals)