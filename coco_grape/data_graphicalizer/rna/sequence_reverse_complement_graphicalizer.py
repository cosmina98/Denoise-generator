import re
import networkx as nx 

def make_reverse_complement_kmer(kmer, complement_map):
    complement_kmer = ''.join([complement_map.get(nt,'X') for nt in kmer])
    reverse_complement_kmer = complement_kmer[::-1]
    return reverse_complement_kmer

def split_kmers(seq, k):
    n = len(seq)
    n_kmers = n//k
    kmer_list = [seq[i*k:(i+1)*k] for i in range(n_kmers)]
    return kmer_list

def find_all_occurrences_of_reverse_complement(kmer, seq, complement_map):
    crkmer = make_reverse_complement_kmer(kmer, complement_map)
    all_starts = [m.start() for m in re.finditer(crkmer, seq)]
    return all_starts

def make_offset_reverse_complement_kmer_graph(orig_seq, k, offset, complement_map):
    seq = orig_seq[offset:]
    kmer_list = split_kmers(seq, k)
    all_occurrences_of_reverse_complement = [find_all_occurrences_of_reverse_complement(kmer, seq, complement_map) for kmer in kmer_list]

    graph = nx.Graph()
    for i,nt in enumerate(seq):
        graph.add_node(i+offset, label=nt)
    for i in range(len(seq)-1):
        #add backbone edges
        graph.add_edge(i+offset, i+1+offset, label='bk', weight=1)

    for i, all_occurrences in enumerate(all_occurrences_of_reverse_complement):
        start = i*k
        for end in all_occurrences:
            for j in range(k):
                u = start + j + offset
                v = end + k - j + offset - 1
                if graph.has_node(u) and graph.has_node(v):
                    if v > u + k: #avoid self interaction of the kmer
                        #add interaction edges to reverse complementary elements
                        graph.add_edge(u, v, label='-', weight= 1 - 1/k)
    return graph

def make_reverse_complement_kmer_graph(orig_seq, k, complement_map):
    graphs = [make_offset_reverse_complement_kmer_graph(orig_seq, k, offset, complement_map) for offset in range(k)]
    out_graph = graphs[0]
    for graph in graphs[1:]:
        out_graph = nx.compose(out_graph, graph)
    return out_graph

def make_reverse_complement_graph(orig_seq, min_k, max_k, complement_map):
    graphs = [make_reverse_complement_kmer_graph(orig_seq, k, complement_map) for k in range(min_k,max_k+1)]
    out_graph = graphs[0]
    for graph in graphs[1:]:
        out_graph = nx.compose(out_graph, graph)
    return out_graph

class SequenceReverseComplementGraphicalizer(object):
    def __init__(self,  min_k, max_k, complement_map = {'A':'U', 'U':'A', 'C':'G', 'G':'C'}):
        self.min_k = min_k
        self.max_k = max_k
        self.complement_map = complement_map
        
    def fit(self, seqs, targets=None):
        return self
    
    def transform(self, seqs):
        graphs = [make_reverse_complement_graph(seq, min_k=self.min_k, max_k=self.max_k, complement_map=self.complement_map) for seq in seqs]
        return graphs
    
    def fit_transform(self, seqs, targets=None):
        return self.fit(seqs, targets).transform(seqs)

