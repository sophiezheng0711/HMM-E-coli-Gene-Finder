p_indel = 10e-8 # probability of insertion and probability of deletion

n_codon = 313

p_gene = 1 - (1 / n_codon) # probability of central state -> one of the 61 codon models

p_interloop = 0.9879

taa_tga = [0.6439, 0.2780]

intergene = [0.2817,0.2219,0.2184,0.2780]

start1_emiss = [0.9063,0.0771,0,0.0166]

p_over = 0.1 # probability of overlapping start/stop codons in comparison to normal start/stop codons